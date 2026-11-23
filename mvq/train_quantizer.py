# Copyright 2026 University of Cambridge (author: Xiaoyu Yang)
# Licensed under the Apache License, Version 2.0; see ../LICENSE.
"""Fit MVQ codebooks from a single embedding CutSet with bounded frame memory."""
import argparse
import json
import logging
import math
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from lhotse import CutSet
from multi_quantization import Quantizer, QuantizerTrainer

from .common import EMBEDDING_DIM, FRAME_SHIFT, check_array, sha256


class ConfigurableTrainer(QuantizerTrainer):
    """Retain the package's two-phase training, with a configurable final size.

    The stock initialization uses 16 entries, then squares to 256. Starting at
    sqrt(target_size) supports 1024 as well, without changing the installed code.
    Based on unified_encoder/local/train_mvq_streaming.py.
    """
    def __init__(self, dim, num_codebooks, codebook_size, phase_one, phase_two, device):
        root = math.isqrt(codebook_size)
        if root * root != codebook_size or codebook_size & (codebook_size - 1):
            raise ValueError('Training codebook size must be a perfect-square power of two')
        self.phase_one_iters, self.phase_two_iters = phase_one, phase_two
        self.cur_iter, self.lr, self.two_iter_prob = 0, 0.005, 0.5
        self.quantizer = Quantizer(dim=dim, num_codebooks=num_codebooks * 2,
                                  codebook_size=root).to(device)
        self.start_time = time.time()
        self._init_optimizer()


def embedding_arrays(manifest, field, dim):
    for cut in CutSet.from_file(manifest):
        yield check_array(cut, field, dim).astype(np.float32, copy=False)


def frame_batches(manifest, field, dim, buffer_frames, batch_size, seed, repeat=True):
    """Shuffle bounded windows; include partial tails instead of dropping frames."""
    rng = np.random.default_rng(seed)
    buffer = np.empty((buffer_frames, dim), dtype=np.float32)
    while True:
        used = 0
        for array in embedding_arrays(manifest, field, dim):
            offset = 0
            while offset < len(array):
                take = min(buffer_frames - used, len(array) - offset)
                buffer[used:used + take] = array[offset:offset + take]
                used += take
                offset += take
                if used == buffer_frames:
                    rng.shuffle(buffer, axis=0)
                    for start in range(0, used, batch_size):
                        yield buffer[start:min(start + batch_size, used)].copy()
                    used = 0
        if used:
            rng.shuffle(buffer[:used], axis=0)
            for start in range(0, used, batch_size):
                yield buffer[start:min(start + batch_size, used)].copy()
        if not repeat:
            break


def channel_stats(manifest, field, dim):
    # Parallel/Welford combination avoids cancellation in sum(x^2) - mean(x)^2.
    count = 0
    mean, m2 = np.zeros(dim), np.zeros(dim)
    for array in embedding_arrays(manifest, field, dim):
        array = array.astype(np.float64)
        local_mean = array.mean(axis=0)
        local_m2 = ((array - local_mean) ** 2).sum(axis=0)
        combined = count + len(array)
        delta = local_mean - mean
        m2 += local_m2 + delta ** 2 * count * len(array) / combined
        mean += delta * len(array) / combined
        count = combined
    if count < 2:
        raise ValueError('Too few frames for normalization')
    std = np.sqrt(m2 / count)
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Normalization requires finite, nonconstant embedding channels')
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--embedding-cuts', type=Path, required=True)
    p.add_argument('--quantizer-path', type=Path, required=True)
    p.add_argument('--embedding-field', default='embedding')
    p.add_argument('--num-codebooks', type=int, default=8)
    p.add_argument('--codebook-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--shuffle-buffer-frames', type=int, default=65536)
    p.add_argument('--num-epochs', type=float, default=1)
    p.add_argument('--phase-one-iters', type=int)
    p.add_argument('--phase-two-iters', type=int)
    p.add_argument('--normalize', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.batch_size < 2 or args.shuffle_buffer_frames < args.batch_size:
        p.error('Require batch-size >=2 and shuffle-buffer-frames >= batch-size')
    if args.shuffle_buffer_frames % args.batch_size:
        p.error('shuffle-buffer-frames must be a multiple of batch-size for epoch-based sizing')
    if not math.isfinite(args.num_epochs) or args.num_epochs <= 0:
        p.error('num-epochs must be positive and finite')
    if args.num_codebooks not in (2, 4, 8, 16, 32) or args.codebook_size not in (4, 16, 64, 256, 1024, 4096, 16384, 65536):
        p.error('Unsupported two-phase codebook count/size')
    if (args.phase_one_iters is None) != (args.phase_two_iters is None):
        p.error('Specify both phase iteration counts, or neither')
    if args.phase_one_iters is not None and min(args.phase_one_iters, args.phase_two_iters) < 1:
        p.error('Phase iteration counts must be positive')
    if args.quantizer_path.exists():
        raise FileExistsError(args.quantizer_path)
    if not torch.cuda.is_available():
        raise RuntimeError('MVQ fitting requires a CUDA node')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.basicConfig(level=logging.INFO)
    ids, total_frames = set(), 0
    for cut in CutSet.from_file(args.embedding_cuts):
        if cut.id in ids:
            raise ValueError(f'Duplicate embedding cut ID: {cut.id}')
        ids.add(cut.id)
        check_array(cut, args.embedding_field, EMBEDDING_DIM, load=False)
        total_frames += getattr(cut, args.embedding_field).shape[0]
    if total_frames < args.batch_size:
        raise ValueError('Embedding corpus is smaller than --batch-size')
    embedding_hash = sha256(args.embedding_cuts)
    sidecar = args.embedding_cuts.with_name('embeddings.meta.json')
    if sidecar.exists():
        collection = json.loads(sidecar.read_text())
        if collection['output_sha256'] != embedding_hash or collection['field'] != args.embedding_field:
            raise ValueError('Embedding sidecar does not describe the supplied manifest/field')
        teacher = collection['teacher']
    else:
        teacher = None
    # QuantizerTrainer.done() requires cur_iter > phase_one + phase_two.
    steps = max(3, math.ceil(total_frames * args.num_epochs / args.batch_size))
    phase_one = args.phase_one_iters if args.phase_one_iters is not None else (steps - 1) // 2
    phase_two = args.phase_two_iters if args.phase_two_iters is not None else steps - 1 - phase_one
    device = torch.device('cuda', 0)
    mean, std = channel_stats(args.embedding_cuts, args.embedding_field, EMBEDDING_DIM) if args.normalize else (None, None)
    gpu_mean, gpu_std = (mean.to(device), std.to(device)) if mean is not None else (None, None)
    trainer = ConfigurableTrainer(EMBEDDING_DIM, args.num_codebooks, args.codebook_size,
                                  phase_one, phase_two, device)
    logging.info('Fitting %d frames; phase counters %d/%d; buffer %d frames',
                 total_frames, phase_one, phase_two, args.shuffle_buffer_frames)
    for array in frame_batches(args.embedding_cuts, args.embedding_field, EMBEDDING_DIM,
                               args.shuffle_buffer_frames, args.batch_size, args.seed):
        frames = torch.from_numpy(array).to(device)
        if gpu_mean is not None:
            frames = (frames - gpu_mean) / gpu_std
        trainer.step(frames)
        if any(not torch.isfinite(v).all() for v in trainer.quantizer.parameters()):
            raise FloatingPointError(f'Nonfinite quantizer parameters at update {trainer.cur_iter}')
        if trainer.done():
            break
    state = {k: v.cpu() for k, v in trainer.get_quantizer().state_dict().items()}
    metadata = dict(teacher=teacher, embedding_manifest=str(args.embedding_cuts.resolve()),
                    embedding_sha256=embedding_hash, frames=total_frames,
                    updates=trainer.cur_iter, num_codebooks=args.num_codebooks,
                    codebook_size=args.codebook_size, frame_shift=FRAME_SHIFT,
                    normalize=args.normalize, seed=args.seed)
    args.quantizer_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.quantizer_path.with_name(f'.{args.quantizer_path.name}.{uuid.uuid4().hex}.tmp')
    torch.save(dict(quantizer=state, mean=mean, std=std, metadata=metadata), temporary)
    temporary.replace(args.quantizer_path)
    print(f'Saved quantizer after {trainer.cur_iter} updates: {args.quantizer_path}')


if __name__ == '__main__':
    main()

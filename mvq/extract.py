# Copyright 2026 University of Cambridge (author: Xiaoyu Yang)
# Licensed under the Apache License, Version 2.0; see ../LICENSE.
"""Extract SPEAR embeddings or MVQ labels into one training-ready CutSet."""
import argparse
import gzip
import hashlib
import json
import logging
import math
import shutil
import sqlite3
import uuid
from pathlib import Path

import numpy as np
import torch
from lhotse import CutSet
from lhotse.features.io import NumpyHdf5Writer
from lhotse.utils import fastcopy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .common import (DEFAULT_MODEL, DEFAULT_QUANTIZER, EMBEDDING_DIM, FRAME_SHIFT,
                     absolute_audio_cut, check_array, encode_frames, load_quantizer,
                     sha256, teacher_identity)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-cuts', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--mode', choices=('labels', 'embeddings'), default='labels')
    p.add_argument('--model-path', type=Path, default=Path(DEFAULT_MODEL))
    p.add_argument('--quantizer-path', type=Path, default=Path(DEFAULT_QUANTIZER))
    p.add_argument('--embedding-layer', type=int, default=-1)
    p.add_argument('--teacher-field', default='codebook_indexes')
    p.add_argument('--embedding-field', default='embedding')
    p.add_argument('--num-codebooks', type=int, default=8)
    p.add_argument('--codebook-size', type=int, default=256)
    p.add_argument('--normalization', choices=('auto', 'none', 'standardize'), default='auto')
    p.add_argument('--num-gpus', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--max-duration', type=float, default=200)
    p.add_argument('--max-cuts', type=int, default=200)
    p.add_argument('--encoding-batch-size', type=int, default=2000)
    p.add_argument('--audio-root', type=Path, default=Path.cwd(),
                   help='Base for relative audio paths in the input CutSet')
    return p


def batches(cuts, max_duration, max_cuts):
    """Bound padded waveform duration and keep every cut, including partial tails."""
    pending, longest = [], 0.0
    for cut in cuts:
        if cut.duration > max_duration:
            raise ValueError(f'{cut.id}: a single cut exceeds --max-duration')
        new_longest = max(longest, cut.duration)
        if pending and (new_longest * (len(pending) + 1) > max_duration or len(pending) >= max_cuts):
            yield pending
            pending, longest = [], 0.0
        pending.append(cut)
        longest = max(longest, cut.duration)
    if pending:
        yield pending


class AudioBatches(IterableDataset):
    def __init__(self, paths, max_duration, max_cuts):
        self.paths, self.max_duration, self.max_cuts = paths, max_duration, max_cuts

    def __iter__(self):
        worker = get_worker_info()
        path = self.paths[worker.id if worker else 0]
        for cuts in batches(CutSet.from_file(path), self.max_duration, self.max_cuts):
            audio = []
            for cut in cuts:
                try:
                    waveform = cut.load_audio()
                except Exception as exc:
                    raise ValueError(f'{cut.id}: audio decode failed: {exc}') from exc
                if waveform.shape[0] != 1 or not np.isfinite(waveform).all():
                    raise ValueError(f'{cut.id}: expected finite mono audio')
                audio.append(torch.from_numpy(waveform[0]))
            yield dict(cuts=cuts, audio=pad_sequence(audio, batch_first=True),
                       audio_lens=torch.tensor([len(x) for x in audio]))


def partition_inputs(args, directory, connection):
    count = args.num_gpus * max(1, args.num_workers)
    paths = [directory / f'input-{i:04d}.jsonl.gz' for i in range(count)]
    writers = [CutSet.open_writer(p) for p in paths]
    connection.execute('CREATE TABLE inputs (id TEXT PRIMARY KEY, ordinal INTEGER, recording TEXT, start REAL, duration REAL, channel INTEGER)')
    total = 0
    try:
        for ordinal, cut in enumerate(CutSet.from_file(args.input_cuts)):
            cut = absolute_audio_cut(cut, args.audio_root)
            try:
                connection.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',
                                   (cut.id, ordinal, cut.recording.id, cut.start, cut.duration, cut.channel))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f'Duplicate input cut ID: {cut.id}') from exc
            writers[ordinal % count].write(cut)
            total += 1
    finally:
        for writer in writers:
            writer.close()
    if total == 0:
        raise ValueError('Input CutSet is empty')
    connection.commit()
    return [paths[r * max(1, args.num_workers):(r + 1) * max(1, args.num_workers)]
            for r in range(args.num_gpus)], total


@torch.no_grad()
def extract_rank(rank, args, paths, directory, identity):
    from .teacher import SpearTeacher
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    teacher = SpearTeacher(str(args.model_path)).to(device).eval()
    quantizer, mean, std = (load_quantizer(args.quantizer_path, args.num_codebooks,
                           args.codebook_size, device, args.normalization, identity)
                           if args.mode == 'labels' else (None, None, None))
    loader = DataLoader(AudioBatches(paths[rank], args.max_duration, args.max_cuts),
                        batch_size=None, num_workers=args.num_workers, pin_memory=True)
    field = args.teacher_field if args.mode == 'labels' else args.embedding_field
    with NumpyHdf5Writer(directory / f'arrays-{rank:04d}') as writer, \
            CutSet.open_writer(directory / f'output-{rank:04d}.jsonl.gz') as manifests:
        processed = 0
        for batch in loader:
            embeddings, lengths = teacher(batch['audio'].to(device), batch['audio_lens'].to(device),
                                          args.embedding_layer)
            for i, cut in enumerate(batch['cuts']):
                frames = embeddings[i, :int(lengths[i])]
                if not len(frames) or frames.shape[1] != EMBEDDING_DIM:
                    raise ValueError(f'{cut.id}: invalid teacher output shape')
                if abs(len(frames) - cut.duration / FRAME_SHIFT) > 5:
                    raise ValueError(f'{cut.id}: teacher frame count disagrees with 50 Hz')
                if args.mode == 'labels':
                    values = encode_frames(quantizer, frames, args.codebook_size,
                                           args.encoding_batch_size, mean, std)
                else:
                    if not torch.isfinite(frames).all():
                        raise ValueError(f'{cut.id}: nonfinite teacher embeddings')
                    values = frames.cpu().numpy().astype(np.float32)
                # Hashed storage keys also work for IDs containing slashes or unusual characters.
                key = hashlib.sha256(cut.id.encode()).hexdigest()
                array = writer.store_array(key=key, value=values, temporal_dim=0,
                                           frame_shift=FRAME_SHIFT, start=cut.start)
                manifests.write(fastcopy(cut, custom={**(cut.custom or {}), field: array}))
                processed += 1
                if processed % 100 == 0:
                    print(f'rank={rank} extracted={processed}', flush=True)
    print(f'rank={rank} complete cuts={processed}', flush=True)


def merge_outputs(connection, paths, output, field, dim):
    """Check exact identity/coverage, then publish in original input order."""
    connection.execute('CREATE TABLE outputs (id TEXT PRIMARY KEY, payload TEXT)')
    for path in paths:
        for cut in CutSet.from_file(path):
            original = connection.execute('SELECT recording,start,duration,channel FROM inputs WHERE id=?',
                                          (cut.id,)).fetchone()
            actual = (cut.recording.id, cut.start, cut.duration, cut.channel)
            if original is None or original[0] != actual[0] or original[3] != actual[3] or \
                    not all(math.isclose(original[i], actual[i], abs_tol=1e-6) for i in (1, 2)):
                raise ValueError(f'{cut.id}: extracted cut does not match the input interval')
            check_array(cut, field, dim, load=False)
            try:
                connection.execute('INSERT INTO outputs VALUES (?,?)', (cut.id, json.dumps(cut.to_dict())))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f'Duplicate extracted cut ID: {cut.id}') from exc
    expected = connection.execute('SELECT COUNT(*) FROM inputs').fetchone()[0]
    actual = connection.execute('SELECT COUNT(*) FROM outputs').fetchone()[0]
    if actual != expected:
        raise ValueError(f'Extraction coverage mismatch: expected {expected}, got {actual}')
    with gzip.open(output, 'wt') as stream:
        for (payload,) in connection.execute('SELECT payload FROM outputs JOIN inputs USING(id) ORDER BY ordinal'):
            stream.write(payload + '\n')
    return actual


def run(args):
    if args.num_gpus < 1 or args.num_workers < 0 or args.max_cuts < 1 or args.encoding_batch_size < 1:
        raise ValueError('Invalid GPU, worker, batch, or encoding chunk count')
    if not math.isfinite(args.max_duration) or args.max_duration <= 0:
        raise ValueError('--max-duration must be positive and finite')
    if not args.teacher_field or not args.embedding_field:
        raise ValueError('Output field names must be nonempty')
    if args.codebook_size < 2 or args.codebook_size > 65536 or args.codebook_size & (args.codebook_size - 1):
        raise ValueError('Codebook size must be a power of two in [2,65536]')
    if args.num_codebooks < 2 or args.num_codebooks & (args.num_codebooks - 1):
        raise ValueError('Codebook count must be a power of two, at least two')
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError('Run extraction on a CUDA node with the requested GPU count')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('SPEAR extraction requires a GPU supporting bf16')
    args.model_path = args.model_path.resolve(strict=True)
    identity = teacher_identity(args.model_path, args.embedding_layer)
    normalization = None
    if args.mode == 'labels':
        # Reject incompatible artifacts before loading the large teacher on any GPU.
        quantizer, mean, std = load_quantizer(args.quantizer_path, args.num_codebooks, args.codebook_size,
                                             'cpu', args.normalization, identity)
        normalization = 'none' if mean is None else 'standardize'
        del quantizer, mean, std
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'{args.mode}.jsonl.gz'
    metadata = args.output_dir / f'{args.mode}.meta.json'
    if output.exists() or metadata.exists():
        raise FileExistsError(f'Completed output already exists: {output}; use a new output directory')
    directory = args.output_dir / f'{args.mode}-arrays-{uuid.uuid4().hex}'
    directory.mkdir()
    try:
        with sqlite3.connect(directory / 'join.sqlite') as connection:
            paths, count = partition_inputs(args, directory, connection)
            if args.num_gpus == 1:
                extract_rank(0, args, paths, directory, identity)
            else:
                torch.multiprocessing.spawn(extract_rank, args=(args, paths, directory, identity),
                                            nprocs=args.num_gpus, join=True)
            temporary = directory / 'combined.jsonl.gz'
            field = args.teacher_field if args.mode == 'labels' else args.embedding_field
            dim = args.num_codebooks if args.mode == 'labels' else EMBEDDING_DIM
            merge_outputs(connection, [directory / f'output-{r:04d}.jsonl.gz' for r in range(args.num_gpus)],
                          temporary, field, dim)
        record = dict(teacher=identity, input_manifest=str(args.input_cuts.resolve()),
                      input_sha256=sha256(args.input_cuts), cuts=count, mode=args.mode,
                      field=field, frame_shift=FRAME_SHIFT, num_gpus=args.num_gpus,
                      num_codebooks=args.num_codebooks if args.mode == 'labels' else None,
                      codebook_size=args.codebook_size if args.mode == 'labels' else None,
                      normalization=normalization,
                      quantizer_path=str(args.quantizer_path.resolve()) if args.mode == 'labels' else None,
                      quantizer_sha256=sha256(args.quantizer_path) if args.mode == 'labels' else None,
                      output_sha256=sha256(temporary))
        temporary_metadata = directory / 'metadata.json'
        temporary_metadata.write_text(json.dumps(record, indent=2) + '\n')
        temporary_metadata.replace(metadata)
        temporary.replace(output)
        # Array files are permanent; intermediate CutSets and the disk join index are not.
        for path in directory.iterdir():
            if path.suffix != '.h5':
                path.unlink()
    except BaseException:
        # Never remove arrays after a successfully published manifest references them.
        if not output.exists():
            metadata.unlink(missing_ok=True)
            shutil.rmtree(directory)
        raise
    print(f'Prepared {count} cuts: {output}', flush=True)


def main():
    logging.basicConfig(level=logging.INFO)
    run(parser().parse_args())


if __name__ == '__main__':
    main()

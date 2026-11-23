# Copyright 2026 University of Cambridge (author: Xiaoyu Yang)
# Licensed under the Apache License, Version 2.0; see ../LICENSE.
"""Artifact contracts shared by the offline preparation stages."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from lhotse.array import TemporalArray
from lhotse.cut import MonoCut
from lhotse.utils import fastcopy

FRAME_SHIFT = 0.02
EMBEDDING_DIM = 1280
DEFAULT_MODEL = '/storage/home/yangxiaoyu/models/spear-xlarge-speech-audio-v2'
DEFAULT_QUANTIZER = '/genai/fsx-project/yangxiaoyu/quantizer/spear-xlarge-v2-layer--1-libri-as500-gigam-cb8.pt'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_identity(model_path, layer):
    root = Path(model_path).resolve(strict=True)
    config = json.loads((root / 'config.json').read_text())
    if config.get('model_type') != 'spear' or config.get('output_downsampling_factor') != 1:
        raise ValueError('Expected a SPEAR checkpoint with nominal 50 Hz output')
    if config.get('causal', False):
        raise ValueError('This example supports the non-streaming SPEAR checkpoint')
    if max(map(int, config['encoder_dim'].split(','))) != EMBEDDING_DIM:
        raise ValueError('Expected 1280-dimensional SPEAR XLarge v2 embeddings')
    weights = sorted(root.glob('*.safetensors')) or sorted(root.glob('pytorch_model*.bin'))
    if not weights:
        raise FileNotFoundError(f'No local model weights under {root}')
    return dict(model_path=str(root), config_sha256=sha256(root / 'config.json'),
                weights_sha256={p.name: sha256(p) for p in weights},
                implementation_sha256=json.loads(Path(__file__).with_name('SOURCE_SHA256.json').read_text()),
                embedding_layer=layer, embedding_dim=EMBEDDING_DIM, frame_shift=FRAME_SHIFT)


def matching_teacher(expected, actual):
    # Paths may change when moving the same immutable model artifact.
    keys = ('config_sha256', 'weights_sha256', 'implementation_sha256', 'embedding_layer', 'embedding_dim', 'frame_shift')
    return all(expected.get(k) == actual.get(k) for k in keys)


def absolute_audio_cut(cut, audio_root):
    if not isinstance(cut, MonoCut) or cut.sampling_rate != 16000 or cut.num_channels != 1:
        raise ValueError(f'{cut.id}: expected a mono 16 kHz MonoCut')
    if not math.isfinite(cut.duration) or cut.duration <= 0:
        raise ValueError(f'{cut.id}: invalid duration')
    sources = []
    for source in cut.recording.sources:
        if source.type != 'file':
            raise ValueError(f'{cut.id}: audio must reference existing local files')
        path = Path(source.source)
        path = path if path.is_absolute() else Path(audio_root) / path
        sources.append(fastcopy(source, source=str(path.resolve(strict=True))))
    return fastcopy(cut, recording=fastcopy(cut.recording, sources=sources))


def check_array(cut, field, dim, integer=False, codebook_size=256, load=True):
    info = (cut.custom or {}).get(field)
    if not isinstance(info, TemporalArray):
        raise ValueError(f'{cut.id}: {field} must be a native TemporalArray')
    if info.temporal_dim != 0 or len(info.shape) != 2 or info.shape[1] != dim:
        raise ValueError(f'{cut.id}: invalid {field} shape {info.shape}')
    if not math.isclose(info.frame_shift, FRAME_SHIFT, abs_tol=1e-6):
        raise ValueError(f'{cut.id}: expected 50 Hz {field}')
    if not math.isclose(info.start, cut.start, abs_tol=1e-6):
        raise ValueError(f'{cut.id}: {field} start must match the extracted cut interval')
    if not load:
        return None
    array = cut.load_custom(field)
    if array.ndim != 2 or array.shape[1] != dim or not len(array):
        raise ValueError(f'{cut.id}: invalid loaded {field} shape')
    if abs(len(array) - cut.duration / FRAME_SHIFT) > 5:
        raise ValueError(f'{cut.id}: {field} length does not match the cut duration')
    if integer:
        if not np.issubdtype(array.dtype, np.integer) or array.min() < 0 or array.max() >= codebook_size:
            raise ValueError(f'{cut.id}: noninteger or out-of-range MVQ labels')
    elif not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError(f'{cut.id}: embeddings must be finite floating-point values')
    return array


def load_quantizer(path, num_codebooks, codebook_size, device, normalization='auto', teacher=None):
    from multi_quantization import Quantizer
    # Local checkpoints contain the package state and optional normalization/provenance.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    state = saved.get('quantizer', saved)
    if tuple(state['centers'].shape) != (num_codebooks, codebook_size, EMBEDDING_DIM):
        raise ValueError('Quantizer centers disagree with codebook count/size or embedding dimension')
    if any(not torch.isfinite(v).all() for v in state.values() if torch.is_floating_point(v)):
        raise ValueError('Quantizer checkpoint contains nonfinite values')
    mean, std = saved.get('mean'), saved.get('std')
    if (mean is None) != (std is None):
        raise ValueError('Quantizer must contain both mean and std, or neither')
    normalized = mean is not None
    if normalization != 'auto' and normalized != (normalization == 'standardize'):
        raise ValueError('Requested normalization disagrees with the quantizer checkpoint')
    if normalized:
        if mean.shape != (EMBEDDING_DIM,) or std.shape != (EMBEDDING_DIM,):
            raise ValueError('Invalid quantizer normalization shapes')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Normalization statistics must be finite with strictly positive std')
        mean, std = mean.float().to(device), std.float().to(device)
    provenance = saved.get('metadata', {}).get('teacher')
    if teacher is not None and provenance is not None and not matching_teacher(provenance, teacher):
        raise ValueError('Quantizer was fitted on different teacher weights/configuration or layer')
    quantizer = Quantizer(dim=EMBEDDING_DIM, num_codebooks=num_codebooks, codebook_size=codebook_size)
    quantizer.load_state_dict(state, strict=True)
    return quantizer.to(device).eval(), mean, std


def encode_frames(quantizer, frames, codebook_size, batch_size, mean=None, std=None):
    if not torch.isfinite(frames).all():
        raise ValueError('Teacher embeddings contain nonfinite values')
    if mean is not None:
        # Match the historical extraction formula; no added epsilon.
        frames = (frames - mean) / std
    if not torch.isfinite(frames).all():
        raise ValueError('Normalized teacher embeddings contain nonfinite values')
    chunks = []
    with torch.no_grad():
        for chunk in frames.split(batch_size):
            indexes = quantizer.encode(chunk.float(), as_bytes=False)
            if indexes.min() < 0 or indexes.max() >= codebook_size:
                raise ValueError('Quantizer returned out-of-range labels')
            chunks.append(indexes.cpu())
    values = torch.cat(chunks).numpy()
    return values.astype(np.uint8 if codebook_size <= 256 else np.uint16)

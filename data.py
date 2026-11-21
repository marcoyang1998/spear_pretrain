"""One prepared LibriSpeech CutSet, offline targets, and reference in-batch mixing."""
import hashlib
import random
from functools import partial
import numpy as np
import torch
from lhotse import CutSet, Fbank, FbankConfig
from lhotse.array import TemporalArray
from lhotse.cut import MonoCut, MixedCut
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.collation import collate_matrices
from lhotse.utils import LOG_EPSILON
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from mixing import mix_with_offset, mix_with_delay_max, _load_mixed_cut_single2, _mix_tokens_single


def validation_cut(cut, seed, fraction):
    digest = hashlib.sha256(f"{seed}:{cut.recording.id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def keep_cut(cut, minimum, maximum):
    return minimum <= cut.duration <= maximum


def training_cut(cut, seed, fraction):
    return not validation_cut(cut, seed, fraction)


def validation_rank(cut, rank, world_size):
    value = int.from_bytes(hashlib.sha256(cut.id.encode()).digest()[:8], "big")
    return value % world_size == rank


def make_validation_loader(cuts, args, specs, rank, world_size):
    # Validation uses disjoint shards with local samplers, avoiding the distributed
    # sampler's repeated tail examples. Empty ranks still join metric reduction.
    shard = cuts.filter(partial(validation_rank, rank=rank, world_size=world_size))
    if next(iter(shard), None) is None:
        return None, None
    return make_loader(shard, args, specs, False, 0, 1)


def load_splits(args):
    # TODO: additional manifests and source-specific teacher routing are future work.
    cuts = CutSet.from_file(args.train_cuts).filter(partial(
        keep_cut, minimum=args.min_cut_duration, maximum=args.max_cut_duration))
    if args.validation_fraction == 0:
        train, valid = cuts, None
    else:
        select = partial(validation_cut, seed=args.seed, fraction=args.validation_fraction)
        valid = cuts.filter(select)
        train = cuts.filter(partial(training_cut, seed=args.seed, fraction=args.validation_fraction))
        if next(iter(valid), None) is None:
            raise ValueError("Validation partition is empty; adjust fraction/seed or disable validation")
    if next(iter(train), None) is None:
        raise ValueError("Training partition is empty after filtering/splitting")
    return train, valid


def validate_cut(cut, specs, load_arrays=False):
    if not isinstance(cut, MonoCut) or cut.sampling_rate != 16000:
        raise ValueError(f"{cut.id}: expected a mono, 16 kHz MonoCut")
    for t in specs:
        if not t.active:
            continue
        if not cut.has_custom(t.field):
            raise ValueError(f"{cut.id}: missing required {t.name} field {t.field}")
        info = getattr(cut, t.field)
        if not isinstance(info, TemporalArray):
            raise ValueError(f"{cut.id}/{t.name}: target must be a native Lhotse TemporalArray")
        if info.temporal_dim != 0 or len(info.shape) != 2 or info.shape[1] != t.num_codebooks:
            raise ValueError(f"{cut.id}/{t.name}: expected [T,{t.num_codebooks}], got {info.shape}")
        if abs(info.frame_shift - 1 / t.frame_rate) > 1e-6:
            raise ValueError(f"{cut.id}/{t.name}: target frame_shift {info.frame_shift} disagrees with {t.frame_rate} Hz")
        if cut.start < info.start - 1e-5 or cut.start >= info.end:
            raise ValueError(f"{cut.id}/{t.name}: target start/range does not cover cut start")
        if load_arrays:
            load_targets(cut, t)


def load_targets(cut, teacher):
    try:
        a = cut.load_custom(teacher.field)
    except Exception as exc:
        raise ValueError(f"{cut.id}/{teacher.name}: failed loading {teacher.field}: {exc}") from exc
    if not np.issubdtype(a.dtype, np.integer) or a.ndim != 2 or a.shape[1] != teacher.num_codebooks:
        raise ValueError(f"{cut.id}/{teacher.name}: invalid target dtype/shape {a.dtype}/{a.shape}")
    if len(a) == 0 or int(a.min()) < 0 or int(a.max()) >= teacher.codebook_size:
        raise ValueError(f"{cut.id}/{teacher.name}: empty targets or ID outside [0,{teacher.codebook_size})")
    if abs(len(a) - cut.duration * teacher.frame_rate) > 5:
        raise ValueError(f"{cut.id}/{teacher.name}: target length {len(a)} does not match cut duration")
    # Convert uint8/uint16 before negative padding and torch.where.
    return torch.from_numpy(a.astype(np.int64, copy=True))


def mix_targets(original, incoming, offset, probability):
    """Reference token replacement in overlap; incoming tokens fill extensions."""
    if offset < 0:
        raise ValueError("Negative mixing offset")
    n, m = len(original), len(incoming)
    result = original.new_full((max(n, offset + m), original.shape[1]), -100)
    result[:n] = original
    overlap = max(0, min(n - offset, m))
    if overlap:
        result[offset:offset + overlap] = _mix_tokens_single(
            original[offset:offset + overlap], incoming[:overlap], probability)
    if offset + m > n:
        start = max(0, n - offset)
        result[offset + start:offset + m] = incoming[start:]
    return result


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, args, specs, training):
        self.args, self.specs, self.training = args, specs, training
        self.extractor = Fbank(FbankConfig(num_mel_bins=128))
        self.hdf5_fix = Hdf5MemoryIssueFix()

    def __getitem__(self, cuts):
        self.hdf5_fix.update()
        cuts = list(cuts.sort_by_duration(ascending=False))
        for c in cuts:
            validate_cut(c, self.specs)
        # The reference limits padded duration; reserve delay budget before mixing.
        limit = self.args.max_duration * self.args.batch_duration_factor
        longest = max(c.duration for c in cuts)
        if self.training and self.args.batch_mixing:
            longest += self.args.mix_delay_max if self.args.batch_mix_mode == "max" else 0
        cuts = cuts[:max(1, int(limit // longest))]
        features, durations, ids = [], [], []
        targets = [[] for _ in self.specs]
        for i, c in enumerate(cuts):
            mixed = c
            if self.training and self.args.batch_mixing and len(cuts) > 1 and random.random() < self.args.mixing_prob:
                j = random.choice([j for j in range(len(cuts)) if j != i])
                snr = random.uniform(self.args.min_snr, self.args.max_snr)
                if self.args.batch_mix_mode == "wavlm":
                    mixed = mix_with_offset(c, cuts[j], snr=snr)
                else:
                    mixed = mix_with_delay_max(c, cuts[j], snr=snr, delay=self.args.mix_delay_max)
            if isinstance(mixed, MixedCut):
                audio, _, probability = _load_mixed_cut_single2(mixed)
            else:
                audio, probability = mixed.load_audio(), 0.0
            features.append(torch.from_numpy(self.extractor.extract(audio, 16000)))
            durations.append(mixed.duration)
            ids.append(c.id)
            for k, t in enumerate(self.specs):
                if not t.active:
                    continue
                tokens = load_targets(c, t)
                if isinstance(mixed, MixedCut) and self.args.token_mixing:
                    track = mixed.tracks[1]
                    validate_cut(track.cut, self.specs)
                    incoming = load_targets(track.cut, t)
                    tokens = mix_targets(tokens, incoming, int(track.offset * t.frame_rate), probability)
                elif isinstance(mixed, MixedCut):
                    # Without token mixing, ignore positions beyond the original target.
                    count = int(mixed.duration * t.frame_rate)
                    tokens = torch.cat((tokens, tokens.new_full((max(0, count - len(tokens)), t.num_codebooks), -100)))
                targets[k].append(tokens)
        batch = dict(inputs=collate_matrices(features, padding_value=LOG_EPSILON),
                     input_lens=torch.tensor([len(f) for f in features]), cut_ids=ids,
                     duration=sum(durations), targets=[], target_lens=[])
        for t, arrays in zip(self.specs, targets):
            batch["targets"].append(pad_sequence(arrays, batch_first=True, padding_value=-100) if t.active else None)
            batch["target_lens"].append(torch.tensor([len(a) for a in arrays]) if t.active else None)
        return batch


def make_loader(cuts, args, specs, training, rank=0, world_size=1):
    sampler = DynamicBucketingSampler(cuts, max_duration=args.max_duration, max_cuts=args.max_cuts,
        num_buckets=args.num_buckets, shuffle=training, drop_last=False,
        buffer_size=max(1000, args.num_buckets * 100), seed=args.seed,
        rank=rank, world_size=world_size, sync_buckets=False)
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(LibriSpeechDataset(args, specs, training), sampler=sampler,
        batch_size=None, num_workers=args.num_workers, worker_init_fn=seed_worker,
        generator=generator, persistent_workers=False)
    return loader, generator

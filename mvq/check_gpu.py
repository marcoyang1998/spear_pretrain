"""Check real SPEAR extraction, parallel coverage, normalization, and training IO."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from lhotse import CutSet

from .common import load_quantizer, encode_frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-dir', type=Path, required=True)
    p.add_argument('--input-cuts', type=Path, required=True)
    p.add_argument('--model-path', required=True)
    args = p.parse_args()
    original = list(CutSet.from_file(args.input_cuts))
    one = list(CutSet.from_file(args.check_dir / 'one/labels.jsonl.gz'))
    two = list(CutSet.from_file(args.check_dir / 'two/labels.jsonl.gz'))
    fitted = list(CutSet.from_file(args.check_dir / 'fit/labels.jsonl.gz'))
    for cuts in (one, two, fitted):
        assert [c.id for c in cuts] == [c.id for c in original]
        assert [(c.start, c.duration, c.recording.id) for c in cuts] == [(c.start, c.duration, c.recording.id) for c in original]
    for first, second in zip(one, two):
        np.testing.assert_array_equal(first.load_custom('codebook_indexes'), second.load_custom('codebook_indexes'))
    state = torch.load(args.check_dir / 'fit/quantizer.pt', map_location='cpu', weights_only=False)
    assert state['mean'] is not None and state['std'] is not None
    assert state['metadata']['updates'] >= 3
    quantizer, mean, std = load_quantizer(args.check_dir / 'fit/quantizer.pt', 8, 256, 'cuda', 'standardize')
    embeddings = list(CutSet.from_file(args.check_dir / 'fit/embeddings.jsonl.gz'))
    for embedding, labels in zip(embeddings, fitted):
        assert embedding.id == labels.id
        frames = torch.from_numpy(embedding.load_custom('embedding')).cuda()
        np.testing.assert_array_equal(encode_frames(quantizer, frames, 256, 2000, mean, std),
                                      labels.load_custom('codebook_indexes'))

    # Check the prepared CutSet with the existing training loader and metadata contract.
    from config import parse_args, teachers
    from data import load_splits, make_loader, validate_cut
    training = parse_args(['--train-cuts', str(args.check_dir / 'fit/labels.jsonl.gz'),
                          '--teacher-routing', 'teacher1', '--num-codebooks', '8,8',
                          '--teacher-fields', 'codebook_indexes,dasheng_codebook_indexes',
                          '--validation-fraction', '0', '--num-buckets', '2',
                          '--max-cuts', '1', '--num-workers', '0'])
    specs = teachers(training)
    for cut in fitted:
        validate_cut(cut, specs, load_arrays=True)
    train, _ = load_splits(training)
    loader, _ = make_loader(train, training, specs, training=False)
    seen = [cut_id for batch in loader for cut_id in batch['cut_ids']]
    assert len(seen) == len(original) and set(seen) == {c.id for c in original}

    # Optional test-only reference; production extraction imports only vendored model code.
    project = Path(__file__).resolve().parents[1]
    reference = project.parent / 'unified_encoder/spear/model.py'
    parity = 'skipped (reference checkout unavailable)'
    if reference.is_file():
        sys.path.insert(0, str(reference.parent))
        spec = importlib.util.spec_from_file_location('reference_spear_teacher', reference)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from .teacher import SpearTeacher
        old = module.SpearEncoder(args.model_path).cuda().eval()
        new = SpearTeacher(args.model_path).cuda().eval()
        # Exercise a nonzero recording-time offset as well as the real source audio.
        cut = original[0].truncate(offset=0.25, duration=2, preserve_id=True)
        audio = torch.from_numpy(cut.load_audio()).cuda()
        lengths = torch.tensor([audio.shape[1]], device='cuda')
        old_frames, old_lengths = old.get_embeddings(audio, lengths)
        new_frames, new_lengths = new(audio, lengths)
        torch.testing.assert_close(new_frames, old_frames, rtol=0, atol=0)
        torch.testing.assert_close(new_lengths, old_lengths, rtol=0, atol=0)
        parity = 'exact outputs and lengths on nonzero-start audio'
    print(json.dumps(dict(cuts=len(original), parallel_labels='exact',
                          normalized_fit='matches encoding of collected embeddings',
                          training_loader='all cuts retained', reference_teacher=parity,
                          passed=True), indent=2))


if __name__ == '__main__':
    main()

"""Verify a prepared label CutSet and all referenced target arrays."""
import argparse
from lhotse import CutSet
from .common import absolute_audio_cut, check_array


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-cuts', required=True)
    p.add_argument('--teacher-field', default='codebook_indexes')
    p.add_argument('--num-codebooks', type=int, default=8)
    p.add_argument('--codebook-size', type=int, default=256)
    args = p.parse_args()
    seen, duration = set(), 0.0
    for cut in CutSet.from_file(args.input_cuts):
        if cut.id in seen:
            raise ValueError(f'Duplicate cut ID: {cut.id}')
        seen.add(cut.id)
        absolute_audio_cut(cut, '.')
        check_array(cut, args.teacher_field, args.num_codebooks, integer=True, codebook_size=args.codebook_size)
        duration += cut.duration
    if not seen:
        raise ValueError('Empty label CutSet')
    print(f'Verified {len(seen)} cuts / {duration:.2f} seconds, {args.num_codebooks} codebooks, size {args.codebook_size}')


if __name__ == '__main__':
    main()

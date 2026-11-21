"""Validate every cut and required target in the single prepared manifest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lhotse import CutSet
from config import parse_args, teachers
from data import validate_cut


if __name__ == "__main__":
    args = parse_args()
    specs = teachers(args)
    ids, count, hours = set(), 0, 0.0
    for cut in CutSet.from_file(args.train_cuts):
        if cut.id in ids:
            raise ValueError(f"Duplicate cut ID: {cut.id}")
        ids.add(cut.id)
        validate_cut(cut, specs, load_arrays=True)
        count += 1
        hours += cut.duration / 3600
    if count == 0:
        raise ValueError("Empty manifest")
    print(f"Verified {count} cuts / {hours:.2f} hours; active teachers: {[t.name for t in specs if t.active]}")

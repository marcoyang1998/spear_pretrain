"""Validation must count every held-out frame once, independent of GPU count."""
import argparse
import ast
from pathlib import Path


def counts(path):
    values = []
    for line in Path(path).read_text().splitlines():
        if "INFO valid step=" in line:
            metrics = ast.literal_eval(line[line.index("{"):])
            values.append({k: v for k, v in metrics.items() if k.endswith("_frames")})
    return values


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("single_gpu_log")
    p.add_argument("ddp_log")
    args = p.parse_args()
    a, b = counts(args.single_gpu_log), counts(args.ddp_log)
    assert a and b and a == b, (a, b)
    print(f"Validation frame counts agree across one/two GPUs: {a[0]}")

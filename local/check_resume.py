"""Compare uninterrupted and resumed runs, including optimizer/scheduler/average state."""
import argparse
import torch


def equal(a, b, location="root"):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=location)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), location
        for key in a:
            equal(a[key], b[key], f"{location}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), location
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, f"{location}[{i}]")
    else:
        assert a == b, f"{location}: {a} != {b}"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("uninterrupted")
    p.add_argument("resumed")
    args = p.parse_args()
    a = torch.load(args.uninterrupted, map_location="cpu", weights_only=False)
    b = torch.load(args.resumed, map_location="cpu", weights_only=False)
    for key in ("model", "model_avg", "optimizer", "scheduler", "grad_scaler", "global_step", "audio_hours"):
        equal(a[key], b[key], key)
    print(f"Exact continuation verified through step {a['global_step']}: model, average, optimizer, scheduler, precision, audio duration")

"""A smoke run must execute optimizer updates, including when using FP16 scaling."""
import argparse
import torch


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    args = p.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    steps = [v["step"] for v in saved["optimizer"]["state"].values() if "step" in v]
    assert steps and min(steps) > 0, "Run completed without optimizer updates"
    print(f"Optimizer updates verified: {min(steps)} steps per parameter batch")

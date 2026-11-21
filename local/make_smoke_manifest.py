"""Synthetic audio/targets for checks only; not a teacher preparation pipeline."""
import argparse
import wave
from pathlib import Path
import numpy as np
from lhotse import CutSet, Recording, NumpyHdf5Writer


def create_manifest(directory, count=24, sizes=(256, 1024), codebooks=(16, 8)):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    cuts = []
    with NumpyHdf5Writer(str(directory / "targets")) as storage:
        for i in range(count):
            duration = 1.2 + (i % 6) * 0.1
            n = int(16000 * duration)
            time = np.arange(n) / 16000
            audio = 0.15 * np.sin(2 * np.pi * (200 + i * 10) * time) + 0.01 * rng.normal(size=n)
            path = directory / f"smoke-{i}.wav"
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes((audio * 32767).astype("<i2").tobytes())
            cut = Recording.from_file(path).to_cut()
            for name, rate, size, cb in zip(("wavlm", "dasheng"), (50, 25), sizes, codebooks):
                targets = rng.integers(size, size=(int(round(duration * rate)), cb),
                                       dtype=np.uint8 if size <= 256 else np.uint16)
                targets[0, :] = size - 1
                setattr(cut, name + "_codebook_indexes", storage.store_array(
                    key=f"{cut.id}-{name}", value=targets, frame_shift=1 / rate,
                    temporal_dim=0, start=0))
            cuts.append(cut)
    manifest = directory / "cuts.jsonl.gz"
    CutSet.from_cuts(cuts).to_file(manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=24)
    args = p.parse_args()
    print(create_manifest(args.output_dir, args.count))

# Five-item SPEAR training example

`librispeech-spear-cb8-example.jsonl.gz` is a native Lhotse CutSet containing
five real LibriSpeech cuts totaling 63.67 seconds from:

`/genai/fsx-project/yangxiaoyu/codebook_indexes/spear-xlarge-v2-cb8-csz256/librispeech-train-all_with_cb.jsonl.gz`

Each cut references mono 16 kHz audio and prepared `codebook_indexes` targets:
eight codebooks, 256 entries per codebook, and a 0.02-second frame shift (50 Hz).
Audio paths are absolute paths to the existing LibriSpeech dataset. Target
storage references and keys point to the existing FSX HDF5 arrays. The audio and
target files must be accessible on the compute node running training.

Run this short single-teacher example from the project root in the existing
`encoder` environment:

```bash
srun -p a100 --gres=gpu:2 --cpus-per-task=8 --mem=24G --time=00:15:00 \
  bash run/train_example.sh
```

[run/train_spear_cb8.sh](../run/train_spear_cb8.sh) lists the full SPEAR teacher,
student, batching, masking/mixing, optimizer, and checkpoint arguments.
[run/train_example.sh](../run/train_example.sh) applies the five-cut overrides:
one GPU, one epoch, five updates, per-step logs, and a checkpoint at step five.
Trailing CLI arguments override either launcher's settings, for example
`bash run/train_example.sh --exp-dir exp/my-example` inside a GPU allocation.

This cluster requires at least two allocated GPUs; the example trains on one.
Validation is disabled because five cuts are too few for the default 1% holdout.
Two duration buckets and one cut per batch ensure all five cuts are sampled.
The installed Lhotse sampler may stop early when choosing empty buckets while
draining partial batches in a tiny dataset; one-cut batches avoid that case.
The inactive second teacher's target field may be absent.

The sample was verified on an A100 compute node by loading all five target
arrays and audio files, checking target/student frame lengths, and collating
all five items through the training data loader. The example launcher also
completed five bf16 optimizer updates in SLURM job 681319; its step-five
checkpoint has finite model and average tensors and records all 63.67 seconds.
The CPU and full GPU smoke suites passed alongside that launcher check.
Only this small manifest and
README are allowed by `.gitignore`; other files under `data/` remain ignored.

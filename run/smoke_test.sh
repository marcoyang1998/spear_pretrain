#!/usr/bin/env bash
# Run on a GPU node, e.g. srun -p a100 --gres=gpu:2 --cpus-per-task=8 --mem=24G --time=00:15:00 bash run/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${ICEFALL_ROOT:-../icefall}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
PYTHON=${PYTHON:-/home/yangxiaoyu/miniconda3/envs/encoder/bin/python}
CHECK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/spear-pretraining-check.XXXXXX")
echo "GPU smoke artifacts: $CHECK_DIR"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" local/make_smoke_manifest.py --output-dir "$CHECK_DIR/data"
"$PYTHON" local/verify_manifest.py --config configs/smoke.json --train-cuts "$CHECK_DIR/data/cuts.jsonl.gz"
"$PYTHON" local/check_learning.py
"$PYTHON" train.py --config configs/smoke.json --train-cuts "$CHECK_DIR/data/cuts.jsonl.gz" --exp-dir "$CHECK_DIR/full"
"$PYTHON" train.py --config configs/smoke.json --train-cuts "$CHECK_DIR/data/cuts.jsonl.gz" --exp-dir "$CHECK_DIR/resumed" --resume "$CHECK_DIR/full/checkpoint-2.pt"
"$PYTHON" local/check_resume.py "$CHECK_DIR/full/checkpoint-6.pt" "$CHECK_DIR/resumed/checkpoint-6.pt"
# A separate test exercises the reference's masked-only option on the 50 Hz grid.
"$PYTHON" train.py --config configs/smoke.json --train-cuts "$CHECK_DIR/data/cuts.jsonl.gz" --exp-dir "$CHECK_DIR/masked" --loss-only-mask 1 --max-iters 2
"$PYTHON" train.py --config configs/smoke.json --train-cuts "$CHECK_DIR/data/cuts.jsonl.gz" --exp-dir "$CHECK_DIR/fp16" --precision fp16 --max-iters 2
"$PYTHON" local/check_optimizer_steps.py "$CHECK_DIR/fp16/checkpoint-2.pt"
"$PYTHON" train.py --config configs/smoke.json --train-cuts "$CHECK_DIR/data/cuts.jsonl.gz" --exp-dir "$CHECK_DIR/workers" --num-workers 2 --max-iters 2
"$PYTHON" train.py --config configs/smoke.json --train-cuts "$CHECK_DIR/data/cuts.jsonl.gz" --exp-dir "$CHECK_DIR/reference-memory-policy" --deterministic 0 --max-iters 2
GPU_COUNT=$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')
if (( GPU_COUNT >= 2 )); then
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 train.py --config configs/smoke.json --train-cuts "$CHECK_DIR/data/cuts.jsonl.gz" --exp-dir "$CHECK_DIR/ddp"
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 train.py --config configs/smoke.json --train-cuts "$CHECK_DIR/data/cuts.jsonl.gz" --exp-dir "$CHECK_DIR/ddp-resumed" --resume "$CHECK_DIR/ddp/checkpoint-2.pt"
  "$PYTHON" local/check_resume.py "$CHECK_DIR/ddp/checkpoint-6.pt" "$CHECK_DIR/ddp-resumed/checkpoint-6.pt"
  "$PYTHON" local/check_validation_counts.py "$CHECK_DIR/full/train-rank0.log" "$CHECK_DIR/ddp/train-rank0.log"
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 train.py --config configs/smoke.json --train-cuts "$CHECK_DIR/data/cuts.jsonl.gz" --exp-dir "$CHECK_DIR/ddp-original" --teacher-routing original --max-iters 2
else
  echo "One GPU available; two-GPU DDP check skipped."
fi
echo "SPEAR PRETRAINING GPU SMOKE CHECKS PASSED"

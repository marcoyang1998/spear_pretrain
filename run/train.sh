#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${ICEFALL_ROOT:-../icefall}:${PYTHONPATH:-}"
PYTHON=${PYTHON:-/home/yangxiaoyu/miniconda3/envs/encoder/bin/python}
if (( ${NUM_GPUS:-1} > 1 )); then
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$NUM_GPUS" train.py --config configs/wavlm_dasheng.json "$@"
else
  "$PYTHON" train.py --config configs/wavlm_dasheng.json "$@"
fi

#!/usr/bin/env bash
# Five optimizer updates on the five-cut example, using the full default student.
# On this cluster: srun -p a100 --gres=gpu:2 --cpus-per-task=8 --mem=24G \
#   --time=00:15:00 bash run/train_example.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export NUM_GPUS=${NUM_GPUS:-1}

exec bash run/train_spear_cb8.sh \
  --train-cuts data/librispeech-spear-cb8-example.jsonl.gz \
  --exp-dir exp/spear-cb8-example \
  --validation-fraction 0 --num-buckets 2 --max-cuts 1 --num-workers 0 \
  --num-epochs 1 --max-iters 5 \
  --save-every-n 5 --average-period 1 --keep-last-k 2 \
  --log-interval 1 \
  "$@"

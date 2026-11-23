#!/usr/bin/env bash
# Run under srun on an A100 node with two allocated GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${ICEFALL_ROOT:-../icefall}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
PYTHON=${PYTHON:-/home/yangxiaoyu/miniconda3/envs/encoder/bin/python}
MODEL_PATH=${MODEL_PATH:-/storage/home/yangxiaoyu/models/spear-xlarge-speech-audio-v2}
QUANTIZER_PATH=${QUANTIZER_PATH:-/genai/fsx-project/yangxiaoyu/quantizer/spear-xlarge-v2-layer--1-libri-as500-gigam-cb8.pt}
INPUT_CUTS=data/librispeech-spear-cb8-example.jsonl.gz
SPEAR_MVQ_CHECK_DIR=$(mktemp -d "$PWD/exp/mvq-check.XXXXXX")
echo "MVQ smoke artifacts: $SPEAR_MVQ_CHECK_DIR"
bash run/smoke_test.sh > "$SPEAR_MVQ_CHECK_DIR/training-smoke.log" 2>&1
tail -n 1 "$SPEAR_MVQ_CHECK_DIR/training-smoke.log"
bash mvq/check_cpu.sh
for spec in one:1:0 two:2:1; do
  IFS=: read -r name gpus workers <<< "$spec"
  "$PYTHON" -m mvq.extract --input-cuts "$INPUT_CUTS" \
    --output-dir "$SPEAR_MVQ_CHECK_DIR/$name" --model-path "$MODEL_PATH" \
    --quantizer-path "$QUANTIZER_PATH" --num-gpus "$gpus" --num-workers "$workers" \
    --max-duration 20 --max-cuts 1
  "$PYTHON" -m mvq.verify --input-cuts "$SPEAR_MVQ_CHECK_DIR/$name/labels.jsonl.gz"
done
# Test the actual stage launcher: collection -> normalized fitting -> live labels -> verification.
STAGE=0 OUTPUT_DIR="$SPEAR_MVQ_CHECK_DIR/fit" INPUT_CUTS="$INPUT_CUTS" \
  MODEL_PATH="$MODEL_PATH" QUANTIZER_PATH="$SPEAR_MVQ_CHECK_DIR/fit/quantizer.pt" \
  NUM_GPUS=1 NUM_WORKERS=0 MAX_DURATION=20 MAX_CUTS=1 \
  QUANTIZER_BATCH_SIZE=128 SHUFFLE_BUFFER_FRAMES=1024 NORMALIZATION=standardize \
  bash mvq/run_spear_xlarge_v2.sh
"$PYTHON" -m mvq.check_gpu --check-dir "$SPEAR_MVQ_CHECK_DIR" \
  --input-cuts "$INPUT_CUTS" --model-path "$MODEL_PATH"
bash run/train_example.sh --train-cuts "$SPEAR_MVQ_CHECK_DIR/fit/labels.jsonl.gz" \
  --exp-dir "$SPEAR_MVQ_CHECK_DIR/student" > "$SPEAR_MVQ_CHECK_DIR/student.log" 2>&1
"$PYTHON" - "$SPEAR_MVQ_CHECK_DIR/student/checkpoint-5.pt" <<'PY'
import sys
import torch
saved = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
assert saved['global_step'] == 5
assert all(torch.isfinite(v).all() for v in saved['model'].values() if v.is_floating_point())
assert all(torch.isfinite(v).all() for v in saved['model_avg'].values() if v.is_floating_point())
print('Five student optimizer updates on newly prepared labels passed.')
PY
echo 'SPEAR MVQ PREPARATION GPU SMOKE CHECKS PASSED'

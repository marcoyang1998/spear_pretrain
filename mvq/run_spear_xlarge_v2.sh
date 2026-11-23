#!/usr/bin/env bash
# Stage 2 reuses the existing quantizer. STAGE=0 collects embeddings and fits a
# new quantizer before extraction. Run from an A100 allocation; no downloads.
set -euo pipefail
if (( $# )); then
  echo 'Override this recipe through environment variables; use python -m mvq.extract for CLI options.' >&2
  exit 2
fi
cd "$(dirname "$0")/.."
export PYTHONPATH="${ICEFALL_ROOT:-../icefall}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
PYTHON=${PYTHON:-/home/yangxiaoyu/miniconda3/envs/encoder/bin/python}
STAGE=${STAGE:-2}
STOP_STAGE=${STOP_STAGE:-3}
INPUT_CUTS=${INPUT_CUTS:-data/librispeech-spear-cb8-example.jsonl.gz}
OUTPUT_DIR=${OUTPUT_DIR:-exp/mvq-spear-xlarge-v2-cb8}
MODEL_PATH=${MODEL_PATH:-/storage/home/yangxiaoyu/models/spear-xlarge-speech-audio-v2}
if (( STAGE <= 1 )); then
  QUANTIZER_PATH=${QUANTIZER_PATH:-$OUTPUT_DIR/quantizer.pt}
else
  QUANTIZER_PATH=${QUANTIZER_PATH:-/genai/fsx-project/yangxiaoyu/quantizer/spear-xlarge-v2-layer--1-libri-as500-gigam-cb8.pt}
fi
NUM_GPUS=${NUM_GPUS:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
NUM_CODEBOOKS=${NUM_CODEBOOKS:-8}
CODEBOOK_SIZE=${CODEBOOK_SIZE:-256}
EMBEDDING_LAYER=${EMBEDDING_LAYER:--1}
TEACHER_FIELD=${TEACHER_FIELD:-codebook_indexes}
MAX_DURATION=${MAX_DURATION:-200}
MAX_CUTS=${MAX_CUTS:-200}
QUANTIZER_BATCH_SIZE=${QUANTIZER_BATCH_SIZE:-1024}
SHUFFLE_BUFFER_FRAMES=${SHUFFLE_BUFFER_FRAMES:-65536}
NORMALIZATION=${NORMALIZATION:-auto}
if (( STAGE < 0 || STAGE > 3 || STOP_STAGE < STAGE || STOP_STAGE > 3 )); then
  echo 'Require 0 <= STAGE <= STOP_STAGE <= 3' >&2
  exit 2
fi
common=(--input-cuts "$INPUT_CUTS" --output-dir "$OUTPUT_DIR" --model-path "$MODEL_PATH"
        --embedding-layer "$EMBEDDING_LAYER" --num-gpus "$NUM_GPUS" --num-workers "$NUM_WORKERS"
        --max-duration "$MAX_DURATION" --max-cuts "$MAX_CUTS")
if (( STAGE <= 0 && STOP_STAGE >= 0 )); then
  "$PYTHON" -m mvq.extract "${common[@]}" --mode embeddings
fi
if (( STAGE <= 1 && STOP_STAGE >= 1 )); then
  normalize=()
  if [[ "$NORMALIZATION" == standardize ]]; then normalize=(--normalize); fi
  "$PYTHON" -m mvq.train_quantizer \
    --embedding-cuts "$OUTPUT_DIR/embeddings.jsonl.gz" --quantizer-path "$QUANTIZER_PATH" \
    --num-codebooks "$NUM_CODEBOOKS" --codebook-size "$CODEBOOK_SIZE" \
    --batch-size "$QUANTIZER_BATCH_SIZE" --shuffle-buffer-frames "$SHUFFLE_BUFFER_FRAMES" \
    "${normalize[@]}"
fi
if (( STAGE <= 2 && STOP_STAGE >= 2 )); then
  "$PYTHON" -m mvq.extract "${common[@]}" --mode labels \
    --quantizer-path "$QUANTIZER_PATH" --num-codebooks "$NUM_CODEBOOKS" \
    --codebook-size "$CODEBOOK_SIZE" --teacher-field "$TEACHER_FIELD" \
    --normalization "$NORMALIZATION"
fi
if (( STAGE <= 3 && STOP_STAGE >= 3 )); then
  "$PYTHON" -m mvq.verify --input-cuts "$OUTPUT_DIR/labels.jsonl.gz" \
    --teacher-field "$TEACHER_FIELD" --num-codebooks "$NUM_CODEBOOKS" --codebook-size "$CODEBOOK_SIZE"
fi

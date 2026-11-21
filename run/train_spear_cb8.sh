#!/usr/bin/env bash
# Single-teacher SPEAR MVQ recipe: 8 codebooks, 256 entries, 50 Hz targets.
# Run inside an A100 allocation. NUM_GPUS defaults to 8; PYTHON and ICEFALL_ROOT
# are handled by run/train.sh. Arguments supplied to this script override the
# recipe below, including --train-cuts, --exp-dir, architecture, and step budget.
# The full manifest's relative audio paths use download/LibriSpeech at this root.
set -euo pipefail
cd "$(dirname "$0")/.."
export NUM_GPUS=${NUM_GPUS:-8}

args=(
  --train-cuts /genai/fsx-project/yangxiaoyu/codebook_indexes/spear-xlarge-v2-cb8-csz256/librispeech-train-all_with_cb.jsonl.gz
  --exp-dir exp/spear-cb8

  # Offline targets: only the first teacher contributes to the loss.
  --teacher-routing teacher1
  --teacher-fields codebook_indexes,dasheng_codebook_indexes
  --num-codebooks 8,8
  --codebook-size 256,256
  --teacher-frame-rate 50,25
  --teacher-loss-weights 1.0,0.0
  --distillation-delta 0,0

  # Original default student: nominal 50 Hz, 64.29M parameters with frontend/mask.
  --num-encoder-layers 2,2,3,4,3,2
  --downsampling-factor 1,2,4,8,4,2
  --encoder-dim 192,256,384,512,384,256
  --encoder-unmasked-dim 192,192,256,256,256,192
  --feedforward-dim 512,768,1024,1536,1024,768
  --num-heads 4,4,4,8,4,4
  --query-head-dim 32 --value-head-dim 12 --pos-head-dim 4 --pos-dim 48
  --cnn-module-kernel 31,31,15,15,15,31
  --output-downsampling-factor 1

  # Duration-based batches and the internal recording-ID validation split.
  --max-duration 200 --max-cuts 200 --batch-duration-factor 5
  --min-cut-duration 0.9 --max-cut-duration 32
  --num-buckets 20 --num-workers 4
  --validation-fraction 0.01 --seed 42

  # Input masking and in-batch waveform/token mixing.
  --mask-mode w2v2 --mask-prob 0.65 --mask-length 10
  --mask-selection static --mask-other 0
  --mask-channel-prob 0.25 --mask-channel-length 15
  --loss-only-mask 0
  --batch-mixing 1 --token-mixing 1 --mixing-prob 0.2
  --batch-mix-mode max --mix-delay-max 2 --min-snr -5 --max-snr 5

  # ScaledAdam/Eden, normal bf16 policy, and training budget.
  --precision bf16 --deterministic 0
  --base-lr 0.045 --warmup-batches 500
  --lr-batches 7500 --lr-hours 30000 --ref-duration 600
  --num-epochs 30 --max-iters 200000

  # Checkpoints include optimizer/resume state and the model average.
  --save-every-n 200 --average-period 100 --keep-last-k 2
  --valid-interval 500 --log-interval 10 --tensorboard 1
)
exec bash run/train.sh "${args[@]}" "$@"

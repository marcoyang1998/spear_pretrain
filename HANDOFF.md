# SPEAR pre-training reproduction: agent handoff

Status checked on 2026-11-20. Implementation and synthetic verification are
complete. No production LibriSpeech pre-training or downstream evaluation has
been performed. The project is a separate initialized Git repository, with all
project files currently untracked/uncommitted. This handoff adds documentation
only; it does not change training code.

**Subsequent update, 2026-11-21:** the user supplied a prepared SPEAR cb8/csz256
LibriSpeech manifest. A four-hour eight-A100 real-data smoke benchmark completed
in job 680170, retaining checkpoint 11800. See VALIDATION.md and
`exp/spear_8gpu_20261120_2152/RESULTS.md`. The original handoff below describes
the state before that benchmark.

## User objective and agreed scope

Create a minimal reproduction of the original SPEAR-style masked MVQ
pre-training recipe at:

`/storage/home/yangxiaoyu/workspace/spear_pretraining`

Primary reference:

`/storage/home/yangxiaoyu/workspace/unified_encoder/zipformer_audio_encoder/train_multi_KD3_shar_speech_audio_multi_mvq2_token_mixing.py`

The user approved planning and implementation, with these decisions:

- Exactly one prepared LibriSpeech Lhotse CutSet `.jsonl.gz`. Additional manifests
  and source-specific routing are future work, noted with a TODO in `data.py`.
- Support two offline teachers and independent distillation deltas.
- Inherit WavLM + Dasheng target metadata; student output must be nominally 50 Hz.
- Keep original backbone, masking, waveform/token mixing, ScaledAdam/Eden,
  mixed precision, DDP, validation, checkpoint averaging and resume.
- Exclude teacher embedding collection, quantizer training, target extraction or
  joining, supervised audio tagging, extra/noise manifests, SHAR, downstream
  fine-tuning/decoding, streaming and export.
- Copy required recipe/model code locally; retain the explicit sibling icefall
  dependency. Runtime must not import unified_encoder.

This reproduces the implementation/objective, not published SPEAR scores or a
specific released checkpoint's complete architecture/data/training budget.

## Read these files first

- `AGENTS.md`: project working instructions.
- `README.md`: current interface, commands, data contract and limitations.
- `REPRODUCTION_PLAN.md`: accepted scope and implementation plan.
- `PROVENANCE.md` and `SOURCE_SHA256.json`: sources and intentional deviations.
- `VALIDATION.md`: completed verification and its limits.

Preserve the five unmodified copied files: `zipformer2.py`, `scaling.py`,
`subsampling.py`, `encoder_interface.py`, `optim.py`. They match recorded source
hashes. Retain original copyright/license notices.

Recorded source checkouts:

- unified_encoder: `1977e3d5e59e087b61a58d95561488119c2302fa`
- icefall: `f7ac574717762d6672060f13228f938c076788bb`

The original unified_encoder repository has a pre-existing modification to
`reports/DIT_TTS_ANALYSIS.md`, which was left untouched. The new project's
untracked `.vscode/` directory was also left untouched during this handoff.

## Teacher, alignment and objective semantics

| Setting | WavLM / teacher1 | Dasheng / teacher2 |
| --- | --- | --- |
| Manifest custom field | `wavlm_codebook_indexes` | `dasheng_codebook_indexes` |
| Number of codebooks | 16 | 8 |
| Entries per codebook | 256 | 256 |
| Native rate | 50 Hz | 25 Hz |
| Native frame shift | 0.02 s | 0.04 s |
| Alignment to student | Direct | Repeat each target frame twice |
| Default delta | 0 | 0 |
| Default loss weight | 1.0 | 0.1 |

Each teacher has a separate `JointCodebookLoss(is_joint=False, reduction="none")`
head. Average cross-entropy over codebooks, then sum valid frames and utterances;
do not change the backpropagated loss to a per-frame mean. Logs report normalized
metrics separately. DDP handles unused parameters because inactive heads and some
package parameters are unused.

**Default routing is `original`: LibriSpeech activates WavLM only.** Both heads
are present, but dual-teacher supervision on the same cuts requires
`--teacher-routing both` and both prepared target fields. Other choices are
`teacher1` and `teacher2`. Target metadata/field names are configurable; names in
logs describe the two roles. Supported teacher rates are initially 25/50 Hz;
the prediction package requires at least two codebooks.

Delta is applied after target alignment to the 50 Hz grid:
`student[t + delta]` predicts `target[t]`. One delta frame is 20 ms. For example,
`--distillation-delta 0,3` gives Dasheng a 60 ms positive offset. This is a temporal
shift, not a feature-difference loss. Categorical targets are repeated, never
interpolated. Align per utterance; use int64 targets and `-100` padding. Reject
large rate/length mismatches and entirely unsupervised batches.

## Default student and augmentation arguments

The student is a six-stack, 16-layer noncausal Zipformer2. It processes mono
16 kHz audio using on-the-fly 128-bin Lhotse fbanks at 100 Hz. Original
Conv2dSubsampling reduces this to nominally 50 Hz; there is no additional output
downsampling. Final embeddings have `max(encoder_dim)=512` channels.

| Architecture CLI argument | Default |
| --- | --- |
| `--num-encoder-layers` | `2,2,3,4,3,2` |
| `--encoder-dim` | `192,256,384,512,384,256` |
| `--encoder-unmasked-dim` | `192,192,256,256,256,192` |
| `--downsampling-factor` | `1,2,4,8,4,2` |
| `--feedforward-dim` | `512,768,1024,1536,1024,768` |
| `--num-heads` | `4,4,4,8,4,4` |
| `--query-head-dim` | `32` |
| `--value-head-dim` | `12` |
| `--pos-head-dim` | `4` |
| `--pos-dim` | `48` |
| `--cnn-module-kernel` | `31,31,15,15,15,31` |
| `--output-downsampling-factor` | `1`, currently the only supported value |

Single stack-parameter values broadcast to all stacks. Internal stack
downsampling does not change the final 50 Hz rate. Fixed construction settings
include noncausal full context, encoder warmup 4000 batches, and frontend/encoder
dropout scheduled from 0.3 at count 0 to 0.1 at count 20,000.

Masking CLI defaults: `mask_mode=w2v2`, `mask_prob=0.65`, `mask_length=10`
(100 ms on the input grid), `mask_selection=static`, `mask_other=0`,
`mask_channel_prob=0.25`, `mask_channel_length=15`, `loss_only_mask=false`.
Block masking is also supported. Masked-only loss projects the input mask onto
the student grid and then applies delta.

Mixing CLI defaults: `batch_mixing=true`, `token_mixing=true`,
`mixing_prob=0.2`, `batch_mix_mode=max`, `mix_delay_max=2` seconds,
`min_snr=-5`, `max_snr=5` dB. Both `max` and `wavlm` modes are supported.
In `both` routing, both active target fields use the same waveform mixture
mapped to each teacher's native grid, with independent stochastic token
replacement. Single-cut batches remain unmixed; no noise manifest is required.

All CLI defaults are in `config.py`. JSON uses underscore keys; CLI overrides
JSON. `configs/wavlm_dasheng.json` specifies teacher defaults and bf16 execution;
`configs/smoke.json` is deliberately small and test-only.

## Input data and trainer

- Input must be a CutSet of mono 16 kHz MonoCuts, not a RecordingSet or
  SupervisionSet. Each active teacher field is a native Lhotse TemporalArray
  `[T, codebooks]` with `temporal_dim=0`, correct frame shift and recording-time
  start. Existing HDF5-backed storage is supported. Both fields must already be
  joined onto the same cut outside this project.
- uint8/uint16 IDs become int64 before padding/mixing/loss. Missing inactive
  fields are allowed; required unreadable, malformed or out-of-range targets fail
  with cut/teacher context. Prefer absolute audio/array storage paths.
- Lazy loading, duration-based bucketing, bounded target loading, and configurable
  cut filtering (default 0.9..32 s). Default batch limits are 200 audio seconds
  and 200 cuts per GPU, with a padded-duration safeguard.
- Default validation is a seeded stable recording-ID hash holdout of 1% from
  the same manifest. It is a monitoring holdout, not official dev/test evaluation.
  Distributed validation uses disjoint shards without duplicated tails.
- ScaledAdam/Eden and original estimated-duration scheduling are preserved.
  Actual consumed audio is tracked separately. Default precision is bf16;
  fp16 uses GradScaler with original initial scale 1.0. fp32 is available.
- Checkpoints are atomic and store model, float64 running average, optimizer,
  scheduler, precision, sampler, per-rank RNG/loader generators, and mutable
  scaling-layer runtime state. Resume checks manifest hash, world size and
  semantic configuration; budget/log/output/retention changes are allowed.

**Resume caveat:** exact continuation is tested with `--deterministic 1
--num-workers 0`. Normal mode retains the original Balancer allocation-dependent
skip heuristic, which prevents guaranteed bitwise continuation across CUDA
processes. Deterministic mode disables only that skip and selects deterministic
CUDA kernels, retaining the regularizer and its probability schedule. Exact
multiworker augmentation resume is not claimed.

## Files implementing the extraction

- `config.py`: CLI/JSON defaults and teacher specs.
- `model.py`: original student construction, independent MVQ heads, alignment,
  delta, routing and objective; `forward_encoder` remains available.
- `masking.py`, `mixing.py`: extracted reference helpers with documented fixes.
- `data.py`: single-manifest IO, target validation, fbanks, batching/mixing and
  holdout/sharding.
- `train.py`, `utils.py`: CUDA/DDP training, schedules, averaging, checkpoints,
  runtime/RNG state and deterministic mode.
- `run/train.sh`, `run/check_cpu.sh`, `run/smoke_test.sh`: launch and verification.
- `local/verify_manifest.py`: production input/target validation only.
- `local/make_smoke_manifest.py`: synthetic test data only, not target preparation.
- Other `local/check_*.py` helpers and `tests/test_contracts.py`: behavior,
  learning, optimizer-update, validation-count and resume checks.

Documented extraction fixes include mutually exclusive precision, seeded masking
and augmentation, NumPy compatibility/short-mask fixes, noise-free mixing,
explicit required-target failures, correct alignment/padding/masked-only deltas,
manifest max-step enforcement and complete checkpoint runtime state.

## Completed verification

See `VALIDATION.md`; tests ran on 2026-11-19 in the existing encoder environment.
Final full smoke SLURM allocation: **677632** (two A100 GPUs). Additional fp16
optimizer-update check: **677634**.

- CPU suite: 16 passed, one CUDA-only check skipped. CUDA suite: all 17 passed.
- Five copied source hashes match. Original encoder bf16 output/length and
  fixed-weight MVQ loss/input-gradient comparisons match exactly.
- Native Lhotse jsonl.gz/HDF5 IO, uint16 IDs through 1023, alignment/deltas,
  masking, mixing and routing checks passed.
- Both heads receive finite gradients. Fixed-batch weighted loss fell from
  1932.1104 to 1642.9524 across eight ScaledAdam updates.
- Single/two-GPU bf16 train/validate/save/average and deterministic step-2-to-6
  resume comparisons passed exactly, including optimizer/scheduler state.
- Single/two-GPU validation counts agree (WavLM 414; Dasheng 408).
- Masked-only loss, two actual fp16 optimizer updates, two-worker loading,
  original Balancer policy, and original-routing DDP passed.
- Python compilation and shell syntax checks passed.

These checks use 24 synthetic recordings and synthetic prepared targets, with a
small model. No production-manifest validation or full-size pre-training claim
follows from them. Do not repeat the full suite solely for this handoff; rerun
appropriate required checks after behavior changes.

## Environment and next work

Run from the new project root; preserve intra-folder imports. Launchers set
`PYTHONPATH=${ICEFALL_ROOT:-../icefall}` and default to:

`/home/yangxiaoyu/miniconda3/envs/encoder/bin/python`

Verified stack: torch/torchaudio 2.8.0+cu128, lhotse 1.33.0, numpy 2.2.6,
h5py 3.16.0, multi_quantization 0.1.1, TensorBoard 2.21.0,
sentencepiece 0.2.2, k2 `1.24.4.dev20260625+cuda12.8.torch2.8.0`.
Training requires CUDA and cluster GPU allocation. No dependencies were installed
or downloaded during implementation; launchers do not install/download anything.

The next substantive step, if requested, is to identify and validate the user's
real prepared LibriSpeech manifest and active targets, then select production
routing/model/batch/budget and launch an appropriate run. The user has not yet
supplied a production manifest path. Do not start a long production job or commit
files merely because a status/handoff was requested.

Example commands (replace placeholders):

```bash
cd /storage/home/yangxiaoyu/workspace/spear_pretraining
export PYTHONPATH=../icefall:${PYTHONPATH:-}
/home/yangxiaoyu/miniconda3/envs/encoder/bin/python local/verify_manifest.py \
  --train-cuts /absolute/path/cuts.jsonl.gz --teacher-routing both

# Run inside an appropriate GPU allocation:
NUM_GPUS=2 bash run/train.sh \
  --train-cuts /absolute/path/cuts.jsonl.gz \
  --exp-dir /absolute/path/experiment \
  --teacher-routing both --distillation-delta 0,3
```

For future behavior changes, follow AGENTS.md and run `bash run/check_cpu.sh`
plus the GPU smoke check under an appropriate allocation. Do not weaken source
parity or exact deterministic-resume assertions to make tests pass.

# spear_pretraining

Minimal extraction of the original SPEAR-style masked MVQ pre-training recipe.
It trains a non-streaming Zipformer2 student at **50 Hz** from **one prepared
LibriSpeech Lhotse CutSet** (`.jsonl.gz`). It retains the original backbone,
scaling kernels, input masking, in-batch waveform/token mixing, independent MVQ
prediction heads, ScaledAdam/Eden, mixed precision, DDP, validation, checkpoint
averaging, and resume.

Teacher embedding collection, quantizer training, target extraction/joining,
supervised audio tagging, fine-tuning, decoding, SHAR, and extra manifests are
outside this project. See [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md) and
[PROVENANCE.md](PROVENANCE.md) for scope and source history.

## Run from this project root

```bash
cd /storage/home/yangxiaoyu/workspace/spear_pretraining
bash run/train.sh \
  --train-cuts /absolute/path/librispeech_with_targets.jsonl.gz \
  --exp-dir /absolute/path/experiment
```

The default configuration follows original speech routing: **LibriSpeech uses
WavLM only**, and Dasheng targets may be absent. To exercise dual-teacher training
on these same speech cuts, provide both target fields and select `both`:

```bash
NUM_GPUS=2 bash run/train.sh \
  --train-cuts /absolute/path/librispeech_with_both_targets.jsonl.gz \
  --exp-dir /absolute/path/dual_teacher_experiment \
  --teacher-routing both --distillation-delta 0,3
```

`NUM_GPUS` defaults to 1; larger values launch one-node `torchrun`. For multiple
nodes, invoke the usual `torchrun --nnodes ... --node-rank ... --nproc-per-node ...`
with `train.py` and the same arguments, setting `PYTHONPATH` to the icefall checkout.
There is no second cluster-specific trainer. SLURM GPU-node execution is required
on this cluster; launchers do not submit full training jobs automatically.

For the prepared SPEAR eight-codebook targets, [run/train_spear_cb8.sh](run/train_spear_cb8.sh)
lists the complete recipe arguments: teacher metadata, student architecture,
batching, masking/mixing, optimizer schedules, and checkpoint retention. It
defaults to the full LibriSpeech SPEAR manifest and eight GPUs, using the settings
from the four-hour benchmark. Its training budget is step/epoch based; the
launcher's arguments do not impose a four-hour wall-time limit. Run it inside an
A100 allocation and pass trailing CLI arguments to override any setting.
The full manifest resolves relative audio paths through `download/LibriSpeech`
at the project root; this workspace already has the dataset symlink.

For the [five-item example manifest](data/README.md), run:

```bash
srun -p a100 --gres=gpu:2 --cpus-per-task=8 --mem=24G --time=00:15:00 \
  bash run/train_example.sh
```

This uses one GPU for five updates, logs every step, and saves a checkpoint at
step five. `run/train.sh` supplies the shared Python/DDP setup; the two example
launchers provide explicit recipe arguments.

`PYTHON` defaults to `/home/yangxiaoyu/miniconda3/envs/encoder/bin/python`;
override it to use another compatible environment. `ICEFALL_ROOT` defaults to
`../icefall`. These source files are local copies: runtime does not import
unified_encoder. The original intra-folder imports are retained.

## Teachers, alignment, and objective

| Setting | WavLM | Dasheng |
| --- | --- | --- |
| Default field | `wavlm_codebook_indexes` | `dasheng_codebook_indexes` |
| Codebooks | 16 | 8 |
| Entries per codebook | 256 | 256 |
| Target frame rate / shift | 50 Hz / 0.02 s | 25 Hz / 0.04 s |
| Alignment to 50 Hz student | Direct | Repeat each categorical frame twice |
| Initial delta | 0 | 0 |
| Loss weight | 1.0 | 0.1 |

The front end computes per-utterance 128-bin Lhotse fbanks at 100 Hz;
Conv2dSubsampling produces nominal 50 Hz outputs, and
`output_downsampling_factor=1` avoids further output downsampling. Finite sequence
lengths include the original front-end boundary losses. Alignment uses each
utterance's true lengths, trims extra teacher frames, ignores padding with `-100`,
and rejects large unexplained discrepancies. Dasheng retains eight prediction
slots per student frame; its targets are repeated, not interpolated or doubled
in the codebook dimension.

Each teacher delta is applied **after alignment**: student frame `t + delta`
predicts target frame `t`. One delta frame is 20 ms. Padding and utterances shorter
than delta contribute no loss. A batch without usable supervision is rejected.

The retained `JointCodebookLoss` uses `is_joint=False`: independent categorical
prediction with cross-entropy averaged over codebooks, then summed over valid
frames and utterances. `both` minimizes `wavlm_loss + 0.1 * dasheng_loss` by default.
Logs report the unweighted teacher losses per valid frame; backpropagation keeps
the reference summed loss. The prediction heads retain the package's parameter
layout, including parameters unused in independent prediction, so DDP enables
unused-parameter handling. `original`/`teacher1` selects the first teacher;
`teacher2` selects the second.

All target metadata and loss weights can be overridden:

```bash
bash run/train.sh --train-cuts /absolute/path/cuts.jsonl.gz \
  --teacher-routing both \
  --teacher-fields teacher_a_indexes,teacher_b_indexes \
  --num-codebooks 8,8 --codebook-size 1024,256 \
  --teacher-frame-rate 50,25 --teacher-loss-weights 1.0,0.2 \
  --distillation-delta 2,3
```

Initial supported target rates are 25/50 Hz, with at least two codebooks per
teacher (the original package requires this). Student output remains 50 Hz.
Teacher names in logs denote the two roles; the actual target fields can refer
to other prepared teacher artifacts. Teacher checkpoints and embedding dimensions
are unnecessary for this offline objective.

## Input manifest contract

Supply a **CutSet**, not a RecordingSet or SupervisionSet. Every eligible input
cut must be a mono 16 kHz `MonoCut`, reference readable audio, and carry each active
teacher's native Lhotse `TemporalArray` in its `custom` fields. Arrays have shape
`[teacher_frames, codebooks]`, temporal dimension 0, correct `frame_shift`, and
`start` consistent with the cut's recording-time interval. HDF5 storage is
supported through Lhotse. Uint8/uint16 IDs are converted to int64 before negative
padding, mixing, or loss computation; floating-point and out-of-range targets
are rejected.

Both teacher fields reside on the **same input cut**. Attach/join targets outside
this project before training. The common single-teacher field `codebook_indexes`
can be selected with `--teacher-fields codebook_indexes,dasheng_codebook_indexes`.
Missing inactive-teacher fields are allowed. Missing/corrupt active targets fail
with cut/teacher context, rather than falling back to dummy supervision.

A [five-cut SPEAR cb8 example](data/README.md) is available under `data/`, with
real audio/target references and a short single-teacher training command.

Audio and target storage paths follow Lhotse's normal resolution from the process
working directory. Prefer absolute paths in prepared manifests so moving from
unified_encoder to this project does not change path interpretation. Targets are
loaded in bounded batches; the target corpus is never concatenated into RAM.
Transcripts and BPE assets are not required.

Validate the prepared manifest before a long run:

```bash
export PYTHONPATH=../icefall:${PYTHONPATH:-}
/home/yangxiaoyu/miniconda3/envs/encoder/bin/python local/verify_manifest.py \
  --train-cuts /absolute/path/cuts.jsonl.gz --teacher-routing both
```

Verification checks duplicate cut IDs, target metadata, storage access, integer
ID ranges, and lengths for all active targets. It does not collect or generate
production targets.

## Training controls

Configuration files use underscore argument names; CLI flags override JSON
defaults. `configs/wavlm_dasheng.json` records the agreed teacher defaults;
the remaining model layer/width and optimizer defaults come from the primary
reference entry point. This default architecture is not asserted to be the
released SPEAR-XLarge checkpoint configuration. `configs/smoke.json` is a small
test-only architecture, with 1024-entry Dasheng targets to exercise uint16 IO.
Use `python train.py --help` for architecture and training-budget options.

- `--max-duration` sets pooled audio seconds per GPU batch; `--max-cuts` bounds
  cut count. A padded-duration safeguard reserves mixing-delay space.
- Cuts are filtered to `[0.9,32]` seconds by default. These limits are configurable.
- A stable seeded recording-ID hash reserves 1% of the single manifest for
  validation; use `--validation-fraction 0` to disable it or another fraction.
  Related cuts remain in the same partition. This is a monitoring holdout, not
  the official LibriSpeech dev/test benchmark. Empty partitions fail early.
  Distributed validation uses disjoint shards, counting each example once even
  when ranks have unequal numbers of batches or an empty shard.
- Time/channel input masking follows the reference w2v2 implementation. Block
  masking is also available. Default loss includes all valid frames; optional
  `--loss-only-mask 1` maps the input mask to the 50 Hz student and then applies delta.
- In-batch speech mixing defaults on (`p=0.2`, SNR -5..5 dB, maximum delay 2 s).
  Choose `--batch-mix-mode wavlm` for original partial-overlap mixing or `max`
  for delay/extension. `--batch-mixing 0` disables waveform mixing; `--token-mixing 0`
  keeps original clean targets for mixed audio and ignores extensions.
  In `both` routing, both target fields use the same waveform overlap mapped to
  their individual frame grids, with independent stochastic codebook replacement.
  Single-cut batches remain unmixed. No external noise manifest is loaded.
- Precision is mutually exclusive: `bf16` (default), `fp16`, or `fp32`.
- LR and model schedules retain the reference estimated audio-duration convention
  `step * max_duration * world_size`, while logs also track actual consumed audio.
- `--max-iters` stops inside an epoch for manifest inputs. Checkpoints contain
  model, float64 running average, optimizer, scheduler, precision state, sampler,
  rank RNG state, loader generator state, and the original scaling layers' mutable
  runtime state (whitening probability, memory-cutoff counters, positional caches,
  schedule counts). Writes are atomic; retention touches
  only numbered checkpoints in the experiment directory.

Resume by supplying the same semantic options/configuration:

```bash
bash run/train.sh --train-cuts /absolute/path/cuts.jsonl.gz \
  --exp-dir /absolute/path/experiment \
  --resume /absolute/path/experiment/checkpoint-4000.pt --max-iters 100000
```

Resume verifies the manifest hash, world size, and model/data/objective settings.
Changing the step/epoch budget, log cadence, output directory, or retention is
allowed. Checkpoints are trusted local Python/NumPy training state and use
`torch.load(weights_only=False)`. Exact continuation is checked with
`--deterministic 1 --num-workers 0`; multiworker prefetch/worker RNG may change augmentation on
resume. Runtime does not promise exact continuation after changing artifacts
under unchanged storage paths.

Normal training retains the original Balancer memory-based skip heuristic, so
allocator/kernel-cache differences can change regularizer decisions after resume
even with workers disabled. `--deterministic 1` selects deterministic CUDA kernels
and disables only that skip heuristic while keeping Balancer's probability
schedule and gradient transformation. It is opt-in because it changes the original
memory-pressure policy. The strict resume smoke comparisons use this mode.
Training releases step-local graphs and gradient lists before checkpointing or
starting another forward.

## Verification and environment

```bash
bash run/check_cpu.sh
srun -p a100 --gres=gpu:2 --cpus-per-task=8 --mem=24G --time=00:15:00 \
  bash run/smoke_test.sh
```

The focused checks cover original loss/gradient parity, masking parity,
25->50 Hz alignment, deltas, padding, short examples, optional fields, true
Lhotse HDF5/uint16 IO, both mixing modes, and configuration/resume validation.
The GPU script additionally checks original encoder parity, fixed-batch learning,
bf16 training/validation, save/resume, masked-only loss, and two-GPU both/original
routing. Source parity checks use the optional sibling unified_encoder checkout;
they skip when that comparison source is unavailable. Production training has no
such dependency. Synthetic audio/targets are used only in smoke tools.

The verified cluster environment is `encoder`: torch/torchaudio 2.8.0+cu128,
lhotse 1.33.0, numpy 2.2.6, h5py 3.16.0, multi_quantization 0.1.1,
TensorBoard 2.21.0, sentencepiece 0.2.2, and
k2 `1.24.4.dev20260625+cuda12.8.torch2.8.0`. The k2 build must match torch/CUDA;
the unchanged activation kernels require its CUDA implementation. The sibling
icefall checkout revision is recorded in PROVENANCE.md. See requirements.txt for
ordinary Python dependencies. No launcher installs packages or downloads models.

Additional manifests/source mixtures, external noise, causal/streaming recipes,
other student output rates, downstream evaluation, and export are future work.

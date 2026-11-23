# Minimal SPEAR-style MVQ pre-training reproduction

Implementation-ready plan, 2026-11-19, incorporating the user's scope decisions.
The project has been implemented according to the initial scope below. The user
extended it on 2026-11-21 to include offline MVQ preparation under `mvq/`: local
SPEAR XLarge v2 embedding collection, optional bounded-memory codebook fitting,
label extraction and a training-ready CutSet. This extension keeps teacher
execution outside the trainer and retains one prepared training manifest.
See mvq/README.md for the added pipeline, and README.md for
commands and VALIDATION.md for verification results. The initial student output rate
is 50 Hz; other architecture and training settings retain reference defaults and
remain configurable. No further scope decisions are required before implementation.

## Objective and reference

Build a small project at `/storage/home/yangxiaoyu/workspace/spear_pretraining`
that trains a Zipformer student from one LibriSpeech Lhotse `CutSet` manifest
(`.jsonl.gz`), using
the original SPEAR-style masked MVQ distillation objective. Required capabilities
are dual-teacher supervision and independent distillation deltas for the teachers.

The primary reference is:

`../unified_encoder/zipformer_audio_encoder/train_multi_KD3_shar_speech_audio_multi_mvq2_token_mixing.py`

Its active dependencies include:

- `model_multi_kd_multi_teacher.py`: student wrapper, masking, prediction heads,
  frame alignment, delta, and auxiliary audio tagging.
- `kd_datamodule3_shar_speech_audio_multi_teacher2_token_mixing.py`: sampling and
  dataset configuration, plus many corpus-specific loaders to remove.
- `dataset_speech_audio_mvq2_token_mixing.py`: audio, features, dual targets, mixing.
- `dataset2_token_mixing.py` and `augmentations.py`: mixing helpers.
- `zipformer2.py`, `scaling.py`, `subsampling.py`, `encoder_interface.py`, `optim.py`:
  the model and optimizer implementation to preserve.
- `multi_quantization.prediction.JointCodebookLoss` and selected icefall helpers.

The primary entry point, model wrapper, dataset, and five model/optimizer files
checked are byte-identical to the corresponding files in
`../icefall/egs/general_audio_encoder/mtl/zipformer_audio_encoder/`.
Current checkout identifiers are unified_encoder
`1977e3d5e59e087b61a58d95561488119c2302fa` and icefall
`f7ac574717762d6672060f13228f938c076788bb`.
Record source file hashes and retain copyright/license notices when extracting.

This is an implementation and objective reproduction. Reproducing a published
SPEAR checkpoint's scores would additionally require its exact teacher artifacts,
data mixture, model configuration, augmentation, and training budget. The inspected
entry point alone does not establish all of those choices.

## Accepted scope

The user accepted these scope choices:

| Decision | Accepted scope |
| --- | --- |
| Data | One LibriSpeech manifest; additional manifests are future work, noted in code comments |
| Dual-teacher routing | Configurable teacher eligibility; default to original speech/audio routing |
| MVQ preparation | Initially excluded; added under `mvq/` by the 2026-11-21 scope extension |
| Token mixing and auxiliary tasks | Keep token mixing; omit supervised audio tagging and downstream fine-tuning |
| Project independence | Copy required recipe/model code locally; retain explicit sibling icefall dependency |
| Teacher configuration | Inherit WavLM + Dasheng target metadata from the original recipe; keep it configurable |
| Student output | 50 Hz, with `output_downsampling_factor=1` after Conv2dSubsampling |

An **example configuration** is a ready-to-run set of training settings: student
architecture, prepared teacher target field names/counts/sizes/frame rates,
routing, loss weights, deltas, masking/mixing settings, and training budget.
It does not load the teacher models or prepare data. The user selected the
WavLM + Dasheng configuration. Separate small/large student presets have not been
requested; teacher metadata does not determine the student's architecture.

### Accepted WavLM + Dasheng target defaults

| Teacher | Target field | Codebooks | Entries per codebook | Target rate | Frame shift | Initial delta | Loss weight |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WavLM | `wavlm_codebook_indexes` | 16 | 256 | 50 Hz | 0.02 s | 0 | 1.0 |
| Dasheng | `dasheng_codebook_indexes` | 8 | 256 | 25 Hz | 0.04 s | 0 | 0.1 |

Codebook counts, deltas, weights, and frame ratios are taken from
`../icefall/egs/general_audio_encoder/mtl/run/train_600m_uniform_v2_zipformer_speech_audio_mvq_bucket_wavlm_all.sh`.
The target rates are the 50/25 Hz defaults in the primary entry point's data module;
the launcher does not override them. The 256-entry size is the default of the
original `JointCodebookLoss`, whose construction does not override codebook size.

For context, at the older launcher's nominal 25 Hz student output rate
(`output_downsampling_factor=2` after the convolutional subsampler), the teacher
frame ratios are `2,1`: WavLM groups two successive 16-codebook frames, giving
32 prediction slots per student frame; Dasheng retains 8 prediction slots.
The accepted reproduction uses nominal 50 Hz student output instead:

- `output_downsampling_factor=1`: the 100 Hz fbank sequence is downsampled by the
  convolutional front end to nominal 50 Hz, with no additional output downsampling.
- WavLM: direct alignment, 16 prediction slots per student frame.
- Dasheng: repeat each 25 Hz target frame twice using nearest-neighbor repetition,
  retaining 8 prediction slots per student frame. Do not interpolate categorical IDs.
- The original head-multiplication setting `teacher_frame_ratio` is `1,1`;
  Dasheng's separate target-repeat factor is 2. Do not mistake the physical
  Dasheng/student rate ratio of 0.5 for an integer head-multiplication factor.
- Apply each teacher's delta after alignment to this 50 Hz student grid. One
  delta frame is 20 ms, and the initial deltas are `0,0`.

Infer/validate alignment from configured rates and actual sequence lengths.
The initial runtime and preset target 50 Hz only; additional student-rate presets
are future work. Reference 25 Hz behavior can still be exercised in isolated
parity checks without expanding the initial runtime interface.

These are defaults for the prepared target artifacts, not properties guaranteed
by the teacher name. Validate codebook counts, frame shifts, and ID ranges against
the input manifest/storage; fail on incompatible artifacts. Codebook size is
explicit configuration even when the highest observed ID is below 255. Since
supervision is offline, teacher embedding dimensions and live teacher checkpoints
are unnecessary for student training.

With LibriSpeech classified as speech, original routing activates only the speech
teacher. Configurable routing can activate both teachers on the same LibriSpeech
cuts when both prepared target fields are present. This distinction must be clear
in the example configuration and documentation.

The initial implementation is non-streaming pre-training from scratch, with
checkpoint resume. Defer causal/streaming recipes, intermediate-layer objectives,
fine-tuning, feature reconstruction, ASR decoding, and HF/ONNX export. Keep normal
encoder access through `forward_encoder` and saved architecture metadata. Specific
checkpoint averaging/export tooling can follow if needed.

## Training behavior to preserve

### Teachers and objective

Teachers supply offline discrete indexes; they do not run inside student training.
Each teacher has its own prediction head, codebook count, codebook size, target
frame rate, alignment policy, nonnegative delta, and loss weight. Teacher names and
manifest field names are configurable rather than tied to `wavlm`/`dasheng`.
Support one teacher as an ablation without maintaining a separate trainer.

The original routing is:

| Data domain | Speech-teacher target | Audio-teacher target |
| --- | --- | --- |
| Speech | Required and used | Unused |
| General audio | Required and used | Required and used |

For that routing, preserve the summed objective:

`L = w_s * (L_speech->speech + L_speech->audio) + w_a * L_audio->audio`

Each term is cross-entropy averaged over the equivalent codebooks and summed
over valid time positions and eligible utterances. Keep
`JointCodebookLoss(is_joint=False, reduction="none")`, as the reference actually
uses it. Do not silently turn on autoregressive codebook prediction or change the
backpropagated loss to a per-frame mean. Report per-valid-frame metrics separately.
Supervised audio tagging is excluded from this reproduction.

### Delta and frame alignment

Match the original meaning after aligning teacher targets to the student grid:

`student[:, delta:, :]` predicts `aligned_targets[:, :-delta, :]` for delta > 0.

For delta 0, retain the full valid sequence. Thus student frame `t + delta`
predicts teacher target `t`; delta is a temporal offset, not a feature-difference
loss. It is expressed in student frames: at the accepted 50 Hz output rate,
delta 3 is about 60 ms. Configure it independently per teacher, e.g.
`--distillation-delta 0,3`.

Preserve the reference's target-alignment semantics while adapting to the accepted
50 Hz output: direct WavLM alignment and explicit 2x Dasheng target repetition.
Use actual per-cut target and encoder lengths; ignore padding
with signed-int64 `-100` targets. Bound small front-end length discrepancies and
reject unexplained rate/length mismatches. An utterance with no positions remaining
after delta contributes no loss; an entirely unsupervised batch is rejected.

`distillation_layer` is stored by the inspected model but does not select an
intermediate output in its loss path. The first reproduction uses the final
encoder output and does not expose this inactive flag as a working feature.

### Model, masking, and training machinery

- Preserve Conv2dSubsampling, Zipformer2, scaling/activation kernels, initialization
  rules, dropout schedules, and the ScaledAdam/Eden implementations.
- Preserve input fbank masking, with the original w2v2 time/channel behavior.
  Retain block masking as a small existing option. Default to loss on all valid
  frames, as the reference does; retain masked-only loss as a disabled-by-default
  option and align its mask
  correctly for output downsampling and delta.
- Compute 128-bin Lhotse fbanks per utterance on the fly, matching the active
  reference path. Start with validated mono, 16 kHz cuts. Transcripts and BPE
  assets are unnecessary for MVQ-only training.
- Retain duration-based batching, lazy loading of the single LibriSpeech manifest,
  and max-step/epoch stopping. Use a deterministic internal train/validation split
  of that one manifest: default 1% validation, configurable or disableable, assigned
  by a stable seeded hash of recording ID so related cuts stay together. This is
  a monitoring holdout, not the official LibriSpeech dev/test evaluation. Check both
  partitions are nonempty when validation is enabled. Defer source mixing/weights and
  additional manifest inputs.
- Retain mixed precision, multi-GPU DDP, scheduled batch counts, checkpoint
  averaging, logs, TensorBoard, periodic saving, retention, and training resume.
- Preserve the original duration/world-size schedule convention; log actual
  consumed audio separately. Do not change LR scheduling while simplifying IO.

Default execution supports one GPU and `torchrun` on one node. Using the same
standard rank/local-rank initialization also permits multiple nodes without a
second cluster-specific training path. GPU validation is required for runtime
claims; CPU checks cover contracts and suitable isolated loss utilities.

Routine defaults do not require further scope discussion: keep the reference
student layer/width defaults and optimizer/schedule defaults, set output
downsampling to 1, choose mutually exclusive bf16 precision by default, and expose
architecture/training-budget overrides. GPU batch duration is configurable and
smoke checks use a small budget. Manifest and experiment paths are supplied when
running; identifying production artifact paths is not a prerequisite to extracting
or testing the project.

## Manifest contract

Input must be a Lhotse **CutSet**, not merely RecordingSet/SupervisionSet JSONL.
A plain audio CutSet is sufficient for audio loading but does not contain the MVQ
supervision by itself.

Prefer native Lhotse `TemporalArray` custom fields pointing to existing storage
(including HDF5), with shape `[teacher_frames, num_codebooks]`, frame shift,
temporal dimension, and start metadata. Two teacher fields can coexist on a cut.
Both uint8/256-entry and uint16/larger-codebook targets are cast to int64 before
padding, token mixing, and cross-entropy. Target IDs must be in the configured
range; an intentionally ineligible teacher may be absent, but required targets
must load successfully.

Provide a validation utility for the single prepared LibriSpeech manifest.
Preparing or joining teacher manifests is outside this project's scope. The
input already contains the configured target fields for every active teacher;
single-teacher unified_encoder outputs commonly use `codebook_indexes`, so a
dual-target input must already attach the second field under a distinct name.
Validate recording/segment identity, target shapes/ranges, storage access, and
cut start/frame metadata. Do not copy large arrays into manifests or load a
complete target corpus into RAM.

Assign the speech domain and configurable teacher eligibility for this single
LibriSpeech manifest. No additional training, validation, or noise manifest input
is implemented initially. Note the extension point in data-loading comments, e.g.
`TODO: support additional manifests and source-specific teacher routing in future.`
Avoid hard-coded corpus sizes, path rewrites, transcript dependencies, and
machine-specific storage clients.

## Token mixing

Extract the original waveform mixing, offset/SNR computation, and stochastic
per-time/per-codebook target replacement helpers. Preserve original mode choices
(`wavlm` overlap and `max` delay), overlap handling, and target extension beyond
the reference utterance.

The original dual-teacher implementation token-mixes speech-teacher targets on
speech examples. General audio receives noise augmentation; its two clean teacher
targets remain unchanged. Keep that behavior for the original-routing preset.
Any broader routing preset must define which eligible teacher fields are mixed;
both contributing cuts must have those targets, with the same waveform overlap
mapped to each teacher's frame grid.

Use in-batch LibriSpeech mixing; external noise-manifest augmentation is deferred
under the one-manifest scope. Preserve the mixing distribution while replacing
wall-clock-dependent random seeding with seeded RNG state. Validation uses no
mixing and no training-time masking.

## Proposed compact layout

```text
spear_pretraining/
  README.md, AGENTS.md, LICENSE, PROVENANCE.md
  train.py                 # extracted and simplified reference training loop
  model.py                 # dual-target wrapper and original objective
  data.py                  # one LibriSpeech manifest; TODO: additional manifests
  masking.py, mixing.py    # extracted reference helpers
  utils.py                 # retained training-specific utilities
  zipformer2.py, scaling.py, subsampling.py, encoder_interface.py, optim.py
  configs/                 # WavLM + Dasheng metadata; configurable student settings
  local/verify_manifest.py
  run/train.sh, run/smoke_test.sh
  tests/                   # focused behavior/parity checks
  requirements.txt
  .gitignore
```

Use a flat training directory so the existing intra-folder imports continue to
resolve. Run launchers from the new project root. Source files are local copies,
not symlinks or imports back into unified_encoder. Under the accepted dependency
choice, launchers supply the explicit icefall path; runtime requirements remain
torch, torchaudio, numpy, lhotse, multi_quantization, k2, and logging dependencies.
Pin the tested environment and record icefall revision. Confirm transitive
dependencies before claiming that removing an unused import removes a package.
No automatic downloads or runtime pip installs.

## Intentional fixes during extraction

Document and verify these separately from scope removals:

1. Mutually exclusive precision selection; the reference defaults both fp16 and
   bf16 to true despite asserting that they cannot both be enabled.
2. Noise-free mixing must work; the reference augmentation unconditionally accesses
   its noise CutSet, including when no noise data was provided.
3. Required target loading must fail with a cut/teacher/storage error rather than
   silently substitute dummy supervision. Optional missing targets remain explicit.
4. Alignment padding must concatenate along time and use explicit padded widths.
   Delta and masked-only loss must use correct lengths at 25 Hz and 50 Hz.
5. Enforce max steps inside manifest-based epochs; the reference's inner-loop
   iteration limit is conditional on SHAR.
6. Remove accidental dependencies on dummy text, auxiliary labels, corpus paths,
   and SHAR internals when those functions are outside the chosen scope.

Fix these only in the new project. The original recipe remains a comparison
reference, including its existing Function-2 variants.

## Implementation sequence and acceptance criteria

1. **Resolve scope and freeze the reference.** Record answers, hashes, dependencies,
   chosen presets, and original-versus-intentional-change matrix.
2. **Extract model and loss.** Copy core model/optimizer files without kernel
   substitutions. Extract masking and dual-head wrapper. Verify fixed-weight
   encoder/loss parity and teacher routing before changing the data pipeline.
3. **Implement manifest IO.** Add one-manifest lazy loading, target validation,
   on-the-fly fbanks, signed target collation, and in-batch mixing. Note additional
   manifest support as future work in comments. Exercise
   true Lhotse serialization and target storage on a tiny synthetic corpus.
4. **Extract the trainer.** Keep the optimizer/scheduler and checkpoint machinery;
   replace corpus/SHAR branches with the LibriSpeech manifest input. Add standard DDP,
   explicit precision, validation, and a step limit that always works.
5. **Verify end to end and document.** Run a short CUDA train/validate/save/resume
   job and a two-GPU test. Publish commands, input schema, configuration, provenance,
   and the retained/removed features. Do not start a full pre-training run as part
   of project extraction.

Focused checks must establish:

- Reference-versus-extracted outputs and loss/gradients with identical loaded
  weights and controlled randomness on supported, well-defined configurations.
- Independent teacher counts/sizes/weights, original and configurable routing,
  missing ineligible targets, and gradients reaching each active prediction head.
- Delta 0 and positive independent deltas, variable lengths, padding, and short
  utterances; 50->25 Hz grouping and 25->50 Hz repetition.
- Codebook IDs above 255 survive uint16 storage, int64 conversion, and mixing.
- Mixing: probability endpoints, offsets, overlap/extension, both teacher
  grids where applicable, single-speech-example batches, and no-noise operation.
- Actual optimizer steps have finite losses/gradients; short-run behavior improves
  on a fixed tiny batch. Two-GPU training handles domain-restricted/unused heads
  without hangs or reduction errors.
- Resume restores model, average, optimizer, scheduler, precision state, global
  step, sampler, relevant RNG state, and mutable scaling-layer runtime attributes. Check deterministic continuation with
  deterministic mode and workers disabled; document multiworker prefetch limits instead of promising
  bitwise-exact continuation under every loader configuration.

Acceptance is a runnable, documented original-style training recipe with verified
dual-teacher and delta behavior, not a copied directory of unrelated entry points
or a claim of paper-level performance from a smoke test.

## Implementation finding: deterministic resume

Normal training retains the original Balancer allocation-dependent skip heuristic.
Its decisions can change across CUDA processes even after RNG and runtime counters
are restored. An opt-in deterministic mode disables only that memory-pressure
skip, retaining the regularizer itself, and selects deterministic CUDA kernels.
Strict resume checks use this mode with zero loader workers. Normal reference-policy
training is checked separately; its resume is not promised bitwise identical.

# Source provenance

Extracted from
`unified_encoder/zipformer_audio_encoder/train_multi_KD3_shar_speech_audio_multi_mvq2_token_mixing.py`
at unified_encoder checkout `1977e3d5e59e087b61a58d95561488119c2302fa`.
The inspected recipe and core modules are byte-identical to
`icefall/egs/general_audio_encoder/mtl/zipformer_audio_encoder/` at icefall checkout
`f7ac574717762d6672060f13228f938c076788bb`.

`zipformer2.py`, `scaling.py`, `subsampling.py`, `encoder_interface.py`, and
`optim.py` are unmodified local copies. SOURCE_SHA256.json records hashes of these
and the source files used for extraction. Original copyright and Apache-2.0 notices
are retained; LICENSE is copied from icefall.

`masking.py` extracts `index_put`, `compute_mask_indices_block`, and
`compute_mask_indices` from `model_multi_kd_w2v2_mask.py`, plus the three masking
methods from `model_multi_kd_multi_teacher.py`. It fixes unseeded per-row NumPy
generators, zero-span inputs, block-mask minimums exceeding available blocks, and
old NumPy `randint`/`np.int` calls. With explicit original seed/epoch/indices and
ordinary supported spans, masking is checked against the reference.

`mixing.py` extracts `mix_with_offset` and `mix_with_delay_max` from
`augmentations.py`, plus energy/probability, `_load_mixed_cut_single2`, and
`_mix_tokens_single` helpers from `dataset2_token_mixing.py`. Waveform/SNR and
replacement-probability calculations are retained. Dataset augmentation uses the
seeded Python RNG and has no noise-dataset dependency. In `both` routing, the
speech-teacher mixing policy is extended to every active teacher field on the
same LibriSpeech waveform mixture.

`model.py` retains the original front-end and Zipformer construction, masking,
`JointCodebookLoss(is_joint=False)`, codebook-normalized/summed objective, and
positive-delta direction. It removes audio-tagging modules and the unused
distillation-layer flag. New alignment explicitly repeats 25 Hz categorical
targets onto the 50 Hz grid, uses per-utterance lengths rather than batch padding,
and excludes student padding before loss at delta zero as well as positive delta.
Masked-only loss uses factor 2 and shifts the projected mask with delta, instead
of the original fixed factor 4. Fixed-weight comparisons are used because removing
the auxiliary head changes fresh-initialization RNG consumption.

`data.py` replaces the large corpus/SHAR datamodule with a single lazy LibriSpeech
CutSet. It retains per-utterance Lhotse fbanks and reference target replacement,
adds explicit teacher metadata/routing, integer conversion, validation and
recording-ID holdout, and rejects unreadable required targets. It respects
TemporalArray's recording-time start/slicing. Extra-manifest support is deliberately
deferred and marked by a TODO in the loader.
Distributed validation uses disjoint cut-ID shards and local samplers, rather
than the training sampler's repeated tail examples; empty shards still reduce
metrics with the other ranks.

`train.py` retains ScaledAdam/Eden, estimated-duration LR/model scheduling,
periodic float64 model averaging, mixed precision, DDP, validation, checkpoint
retention and resume. It simplifies the training loop to one manifest, removes
unused ASR/BPE/auxiliary-task inputs and cluster storage clients, makes precision
mutually exclusive, and enforces max steps inside manifest epochs. BF16 uses
autocast without unnecessary loss scaling; FP16 retains GradScaler with the
original `init_scale=1.0`. Atomic
checkpoints add rank RNG, loader generator state, scaling-layer runtime state
(Whiten probabilities, Balancer memory-cutoff counters, positional caches and
schedule counts), and semantic resume checks. These attributes are absent from
the original model state_dict and affect training continuation.
Step-local graphs/gradient lists are released explicitly to avoid retained memory.
Normal training preserves the original allocation-dependent Balancer skip policy;
that policy prevents guaranteed bitwise continuation across cold CUDA processes.
An opt-in deterministic mode selects deterministic CUDA kernels and disables only
this memory-pressure skip, preserving the probability schedule and regularizer.
Strict continuation comparisons use deterministic mode with workers disabled;
ordinary reference-memory-policy training is checked separately.

WavLM/Dasheng codebook counts, weights, and initial deltas come from
`icefall/egs/general_audio_encoder/mtl/run/train_600m_uniform_v2_zipformer_speech_audio_mvq_bucket_wavlm_all.sh`;
50/25 Hz target rates and 256-entry codebook size are the source module defaults.
The reproduction changes student output from that launcher's 25 Hz to the user's
selected 50 Hz. Model layer/width defaults follow the primary entry point, not an
assertion about the architecture or training budget of a particular released
SPEAR checkpoint.

This is an implementation/objective reproduction. No published downstream scores
or released-checkpoint reproduction are claimed by synthetic smoke verification.

## Offline SPEAR MVQ preparation, added 2026-11-21

The user extended the initial scope to include preparation under `mvq/`.
`mvq/spear_hf/` vendors `configuration_spear.py`, `modeling_spear.py`,
`spear_model.py`, `spear_modules.py`, and `zipformer.py` byte-for-byte from
`/storage/home/yangxiaoyu/models/spear-xlarge-speech-audio-v2`, the local published
[marcoyang/spear-xlarge-speech-audio-v2](https://huggingface.co/marcoyang/spear-xlarge-speech-audio-v2)
artifact. Hashes are in `mvq/SOURCE_SHA256.json`; existing notices remain intact.
The root backbone/optimizer copies remain unchanged.

The reference preparation workflow is `unified_encoder/spear/model.py`,
`spear/collect_embeddings.py`, `spear/extract_mvq.py`, and
`local/train_mvq_streaming.py`. New code retains bf16 teacher compute, float32
embeddings, 50 Hz TemporalArrays, the package codec with `as_bytes=False`, and
the historical `(embedding - mean) / std` normalization. The optional configurable
trainer retains the package's two-phase algorithm with the reference's
sqrt(codebook_size) initialization. The new streaming trainer preserves float32
instead of staging embeddings in float16 and uses a bounded shuffle buffer.

Intentional preparation changes: strict local-only model loading; checkpoint
shape/normalization/provenance validation; finite-value checks; absolute audio
paths; exact duplicate/identity/coverage checks; duration batching that includes
every partial tail; per-rank HDF5 writers; disk-backed joining in input order;
atomic final manifests; and model/config/code/quantizer fingerprints. The pipeline
accepts one CutSet and does not implement corpus mixing or external teacher joining.
The five-cut fitting example is a functional check and does not reproduce the
historical quantizer's LibriSpeech + AudioSet + GigaSpeech training population.

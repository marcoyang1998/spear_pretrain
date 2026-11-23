# SPEAR XLarge v2 MVQ preparation

This folder prepares offline MVQ supervision for the trainer. It supports a
single mono 16 kHz Lhotse CutSet as input and produces one CutSet referencing
native HDF5-backed `TemporalArray` labels. All code and launchers for teacher
loading, embedding collection, quantizer fitting, extraction, and verification
live here. The trainer continues to read prepared targets without loading teachers.

The example uses [SPEAR XLarge speech/audio v2](https://huggingface.co/marcoyang/spear-xlarge-speech-audio-v2):
top-layer `encoder_out` (`--embedding-layer -1`), 1280 dimensions, nominal 50 Hz.
The existing example quantizer has eight codebooks, 256 entries and no
normalization. Its labels are uint8 arrays of shape `[frames,8]`.

## Extract with the existing quantizer

Run from the project root with the existing `encoder` environment. The local
model must contain `config.json` and weights; loading uses `local_files_only=True`.
No launcher installs dependencies or downloads model artifacts.

```bash
srun -p a100 --gres=gpu:2 --cpus-per-task=8 --mem=32G --time=00:15:00 \
  bash mvq/run_spear_xlarge_v2.sh
```

Defaults use the five-item example CutSet, one GPU, and these existing artifacts:

```text
MODEL_PATH=/storage/home/yangxiaoyu/models/spear-xlarge-speech-audio-v2
QUANTIZER_PATH=/genai/fsx-project/yangxiaoyu/quantizer/spear-xlarge-v2-layer--1-libri-as500-gigam-cb8.pt
OUTPUT_DIR=exp/mvq-spear-xlarge-v2-cb8
```

Output:

```text
labels.jsonl.gz                 # training-ready cuts, in original input order
labels.meta.json                # model/config/code/quantizer hashes and settings
labels-arrays-<run-id>/*.h5      # labels referenced by the CutSet; retain this folder
```

The example already has an old `codebook_indexes` field. Extraction replaces
that field in the new output cuts while preserving input files, cut identity,
supervisions and other custom fields. Inputs without targets work the same way.
The field name is configurable. Audio paths in output cuts become absolute;
relative input audio paths resolve from the project root, or `--audio-root`.

For the full LibriSpeech input, allocate eight GPUs and override the recipe's
environment variables. The existing manifest resolves audio through the local
`download/LibriSpeech` symlink:

```bash
srun -p a100 --gres=gpu:8 --cpus-per-task=64 --mem=128G --time=02:00:00 \
  env NUM_GPUS=8 NUM_WORKERS=4 \
  INPUT_CUTS=/genai/fsx-project/yangxiaoyu/codebook_indexes/spear-xlarge-v2-cb8-csz256/librispeech-train-all_with_cb.jsonl.gz \
  OUTPUT_DIR=exp/mvq-spear-librispeech \
  bash mvq/run_spear_xlarge_v2.sh
```

## Collect embeddings and fit a new quantizer

`STAGE=0` runs collection, fitting, label extraction and verification. This
fits on the supplied CutSet; a five-cut fit is a functional example, not a
replacement for the existing quantizer trained on a large speech/audio corpus.

```bash
srun -p a100 --gres=gpu:2 --cpus-per-task=8 --mem=32G --time=00:15:00 \
  env STAGE=0 NUM_GPUS=1 NUM_WORKERS=0 \
  OUTPUT_DIR=exp/mvq-fit-example \
  QUANTIZER_BATCH_SIZE=128 SHUFFLE_BUFFER_FRAMES=1024 \
  bash mvq/run_spear_xlarge_v2.sh
```

Additional outputs are `embeddings.jsonl.gz`, `embeddings.meta.json`,
`embeddings-arrays-<run-id>/*.h5`, and `quantizer.pt`. Embeddings are stored as
float32, preserving the teacher's output precision. Fitting streams through a
bounded shuffled frame buffer; the default 65,536 × 1280 float32 buffer occupies
320 MiB, plus a batch and one cut. It never concatenates the embedding corpus
into RAM. Phase lengths default to approximately one pass over the input frames;
explicit phase counters are available through the Python CLI. The package's
two-phase algorithm is retained. Training sizes must be perfect-square powers
of two, including 256 and 1024.

Set `NORMALIZATION=standardize` when fitting to compute channel statistics in
a streaming pass and save them with the codebooks. Extraction's default `auto`
uses the checkpoint's statistics, applying exactly `(embedding - mean) / std`.
Explicit `none`/`standardize` requests that disagree with the checkpoint fail.
Newly fitted quantizers also verify teacher weights, model configuration, source
implementation and layer on extraction. Legacy quantizers without teacher
provenance are accepted after shape, finite-value and normalization checks.

| Stage | Operation |
| --- | --- |
| 0 | Collect teacher embeddings |
| 1 | Fit quantizer from `embeddings.jsonl.gz` |
| 2 | Extract labels live from audio with the selected quantizer |
| 3 | Verify every label array and audio reference |

`STAGE` defaults to 2; `STOP_STAGE` defaults to 3. Completed outputs are kept
and rerunning extraction into the same completed output directory fails. Choose
a new `OUTPUT_DIR`, or use stage 3 to verify existing output. Interrupted passes
must be restarted; partial-pass resume is not implemented.

The recipe exposes `INPUT_CUTS`, `OUTPUT_DIR`, `MODEL_PATH`, `QUANTIZER_PATH`,
`NUM_GPUS`, `NUM_WORKERS`, `NUM_CODEBOOKS`, `CODEBOOK_SIZE`, `EMBEDDING_LAYER`,
`TEACHER_FIELD`, `MAX_DURATION`, `MAX_CUTS`, `QUANTIZER_BATCH_SIZE`,
`SHUFFLE_BUFFER_FRAMES`, and `NORMALIZATION`. Use the Python CLIs for additional
options:

```bash
/home/yangxiaoyu/miniconda3/envs/encoder/bin/python -m mvq.extract --help
/home/yangxiaoyu/miniconda3/envs/encoder/bin/python -m mvq.train_quantizer --help
/home/yangxiaoyu/miniconda3/envs/encoder/bin/python -m mvq.verify --help
```

## Train on the prepared output

```bash
srun -p a100 --gres=gpu:2 --cpus-per-task=8 --mem=24G --time=00:15:00 \
  bash run/train_example.sh \
  --train-cuts exp/mvq-spear-xlarge-v2-cb8/labels.jsonl.gz \
  --exp-dir exp/train-prepared-example
```

For full-corpus training, use `run/train_spear_cb8.sh --train-cuts ...` inside
the appropriate GPU allocation. Changing the output field, codebook count or
size also requires matching teacher arguments in the trainer.

## Verification and provenance

```bash
bash mvq/check_cpu.sh
srun -p a100 --gres=gpu:2 --cpus-per-task=8 --mem=48G --time=00:15:00 \
  bash mvq/smoke_test.sh
```

Checks cover raw and normalized package-codec parity, IDs above 255, native
HDF5 slicing with nonzero cut starts, complete input coverage and original order,
bounded frame streaming, and rejection of duplicate IDs, missing cuts, incorrect
intervals, normalization mismatches and teacher mismatches. The GPU suite uses
the real local SPEAR model, existing quantizer and five-item audio sample. It
checks one/two-GPU label parity, normalized collection/fit/extraction, optional
reference-teacher parity, and five student optimizer updates on the new labels.
All checks passed in SLURM job 681359 on 2026-11-21. See
[VALIDATION.md](../VALIDATION.md) for results and artifact paths.

`spear_hf/` contains five unchanged source files from the local published model
artifact. Their hashes are recorded in `SOURCE_SHA256.json`; original notices
are retained. See [the project provenance](../PROVENANCE.md). Runtime uses these
local files and never imports `unified_encoder`. Optional preparation dependencies
are listed in `requirements.txt`; they are already present in the encoder environment.

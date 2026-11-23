# Working on spear_pretraining

Run commands from this project root. The copied model files use intra-folder
imports; launchers supply `PYTHONPATH=${ICEFALL_ROOT:-../icefall}`.

Read README.md and REPRODUCTION_PLAN.md before changing scope. Initial data input
is exactly one prepared LibriSpeech CutSet jsonl.gz. Offline teacher embedding
collection, MVQ fitting, label extraction and preparation live under `mvq/`,
using SPEAR XLarge v2 as the example; keep teacher loading out of the trainer.
Additional training manifests, auxiliary supervised tasks and downstream export
are future work. Student output is 50 Hz: WavLM targets align directly, Dasheng targets
repeat twice; delta is applied afterward in 20 ms student frames.

Preserve the unmodified backbone/optimizer copies listed in SOURCE_SHA256.json.
Keep source notices and document intentional deviations in PROVENANCE.md. Runtime
must not import unified_encoder. Reference parity checks may use that sibling
checkout as an optional test-only dependency.

Use the existing encoder conda environment. Training requires a CUDA GPU; on this
cluster run the smoke script under srun on an A100 node. Do not install/download
inside launchers. Run `bash run/check_cpu.sh` and the GPU smoke check for changes to
model/loss/data/training behavior. Keep deterministic resume checks meaningful;
do not remove assertions merely to pass them.

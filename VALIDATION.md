# Verification results

Verified 2026-11-19 in the existing `encoder` environment. The final full GPU
smoke run used SLURM allocation **677632**, two CUDA GPUs on the `a100` partition;
an additional FP16 optimizer-update check passed in allocation **677634**.

| Check | Result |
| --- | --- |
| CPU unittest suite | 17 checks: 16 passed, CUDA-only encoder check skipped |
| CUDA unittest suite | All 17 passed |
| Copied backbone/optimizer hashes | All five files match the recorded source hashes |
| Original encoder comparison | Exact output/length match under bf16; 207/157 input frames -> 100/75 output frames |
| Original MVQ objective comparison | Exact fixed-weight loss and input-gradient match for independently shifted teachers |
| Native Lhotse input/targets | Synthetic audio CutSet jsonl.gz and HDF5 arrays; uint16 IDs through 1023 round-trip correctly |
| Fixed-batch learning | Both prediction heads receive finite gradients; weighted loss decreased from 1932.1104 to 1642.9524 in eight ScaledAdam updates |
| Single-GPU bf16 run | Train, held-out validation, average/checkpoint saving, and six-step limit passed |
| Single-GPU deterministic resume | Resume from step 2 through step 6 exactly matches uninterrupted weights, average, optimizer, scheduler, precision state and consumed audio duration |
| Two-GPU bf16 DDP run | Both-teacher training, validation and six-step limit passed |
| Two-GPU deterministic resume | Same exact continuation comparison passed independently |
| Distributed validation accounting | One/two-GPU counts agree: WavLM 414 valid frames, Dasheng 408; no repeated tail examples |
| Original routing | Two-GPU speech-only/WavLM routing works with an unused Dasheng head |
| Masked-only objective | Two-step 50 Hz CUDA run with independent deltas passed |
| FP16 | Two-step train/validate/save run passed; two optimizer updates per parameter batch verified |
| Multiworker loading | Two-worker train/validate/save run passed |
| Original memory-pressure policy | Non-deterministic two-step both-teacher run passed |
| Static checks | Python compilation and shell syntax checks passed |

Strict continuation checks use `--deterministic 1 --num-workers 0`. Normal
training keeps the original Balancer's allocation-dependent skip heuristic,
which can change regularizer decisions across processes; bitwise resume is not
claimed for that mode or multiworker augmentation. The deterministic option
disables only this skip heuristic and selects deterministic CUDA kernels.

The verification above used 24 synthetic recordings and prepared synthetic targets, not a
production LibriSpeech training run. The smoke configuration is deliberately
small; no full pre-training job, teacher preparation, or downstream benchmark
was performed. Run commands are documented in README.md and run/smoke_test.sh.

## Additional real-data benchmark, 2026-11-20/21

A four-hour, single-teacher SPEAR MVQ smoke run completed on **eight A100 80 GB
GPUs**, SLURM job **680170** (`COMPLETED`, exit `0:0`). The user's prepared
LibriSpeech manifest contains `codebook_indexes` with eight codebooks, 256 entries
and a 0.02-second frame shift. All 281,241 cuts / 961.0544 hours passed target and
audio-path preflight checks; 32 decoded samples matched student frames exactly.
The existing CPU and GPU smoke suites also passed again.

| Check / metric | Result |
| --- | --- |
| Real-data execution | 4h00m01s, bf16, 8-GPU DDP, four loader workers/rank |
| Last logged step | 11,810 |
| Training loss per valid frame | 7.6699 → 2.4950 |
| Held-out validation loss | 7.4887 → 1.5208; best 1.5139 at step 11,000 |
| Post-step-100 mean step time | 1.2146 s, including IO/DDP/validation/checkpoint overhead |
| Actual audio throughput, all eight GPUs | 1,235.8 audio seconds / wall second |
| Numerical failures / OOMs | None observed |
| Latest checkpoint | Step 11,800; finite model and float64 average, eight rank states present |
| Retention | Two checkpoints, approximately 2.88 GB total; no partial checkpoint |
| Preserved source hashes | All five still match |

The four-hour timeout intentionally recorded exit code 124 and retained the
latest periodic checkpoint; the batch job completed successfully. Audio throughput
counts processed waveforms, including mixing extensions and repeated epochs.
This run used the default 64.29M student (70.85M total with prediction heads),
a 200-second per-GPU batch budget, and the internal 1% monitoring holdout.
It is a real-data smoke benchmark without downstream or published-score claims.

The previous plain SPEAR MVQ experiment in `../unified_encoder/spear_gen/` used
the same manifest path, a larger 93.29M student and a 600-second per-GPU budget.
Its first four hours measured 1.3117 s/step and an estimated 3,493.8 audio seconds
per wall second. Architecture, masking, batching, loss accounting, validation and
bf16 scaling differ, so this is a descriptive comparison.

See [the experiment report](exp/spear_8gpu_20261120_2152/RESULTS.md) for configuration,
checkpoint inspection, metrics, curves and comparison details.

## Example launchers, 2026-11-21

`run/train_spear_cb8.sh` records the explicit single-teacher recipe arguments;
`run/train_example.sh` selects the five-item real-data sample for five updates.
Shell syntax checks and the CPU suite passed. SLURM job **681319** passed the
full GPU smoke suite and ran the example launcher with the default student on
one A100 GPU. It processed all five cuts / 63.67 seconds and saved
`checkpoint-5.pt` with finite model and running-average tensors. The output
directory override was honored, and all five preserved source hashes matched.
Artifacts are under `exp/example-launcher-check.KQn6xX/`.

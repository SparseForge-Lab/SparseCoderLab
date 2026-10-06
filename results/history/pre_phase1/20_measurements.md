# Recorded measurements and readiness — 2026-10-05

PASS: phase-0 engineering readiness

No long training run was launched. The tracked goal covers building/verifying this lab; architecture research remains open.

## Built

Native Windows isolated venv and dependency lock; frozen 32,768 byte-BPE; bounded licensed synthetic shards and streaming/import tools; three shared-SDPA baselines; Top1/Top2 MoE; optional N-gram memory; shared/independent three-step MTP; checkpointable BF16 training, timer and batch finder; trace/atlas/packing; three-tier cache/prefetch and scaling simulators; RouteAhead feature/training/evaluation; archived repeated-compaction fixtures and neural NLL evaluator; large-model capacity sweep; disposable toy-agent trajectories. Separate documentation is under Documentation/.

## Environment

GPU: NVIDIA GeForce RTX 5070, capability [12, 0], driver 616.64, 12,227 MiB driver-visible VRAM. Project torch 2.14.1+cu130, wheel CUDA 13.0; arch list ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']. BF16 SDPA backward executed: True. TF32 supported. No silent CPU fallback.

Windows 11; Python 3.12.10; AMD Ryzen 7 5700X (8 cores/16 threads); 51,442,139,136 installed RAM bytes (~47.91 GiB). WSL2 cannot start because virtualization is disabled. No existing installation was patched. D: is a FIKWOT FS810 SATA SSD, not NVMe.

Unbuffered 256MiB file / 4MiB blocks: sequential 462.17 MiB/s; shuffled large-block 461.91 MiB/s. Random block mean 8.660 ms includes transfer; do not double-count as independent latency. Windows page cache bypassed; device cache remains possible.

## Exact parameters

| Model | Stored | Experts | Resident | Memory | MTP | Active estimate |
|---|---:|---:|---:|---:|---:|---:|
| DenseCompute | 16,574,400 | 0 | 16,574,400 | 0 | 0 | 16,574,400 |
| DenseSize | 25,974,720 | 0 | 25,974,720 | 0 | 0 | 25,974,720 |
| SparseV3 | 21,562,560 | 5,529,600 | 16,032,960 | 0 | 0 | 16,493,760 |
| SparseV3Ngram | 25,767,425 | 5,529,600 | 16,032,960 | 4,204,865 | 0 | 16,504,353 |
| SparseV3MTP | 21,921,280 | 5,529,600 | 16,032,960 | 0 | 358,720 | 16,852,480 |
| SparseV3Top2 | 21,562,560 | 5,529,600 | 16,032,960 | 0 | 0 | 16,954,560 |

SparseV3 without memory is 21.56M; the ~25.7M size includes the memory table. Active estimates include the tied dense LM output head and are not measured FLOPs. Shared MTP executes three times even though weights are counted once.

## Actual tests and short smokes

30 tests executed; 0 failures/errors in final JUnit report. Phase-0 gate passed: True. DenseCompute tiny overfit: 10.415672 -> 0.00017485 over 80 steps. CPU and GPU exact checkpoint continuation, data cursor and speculative greedy parity passed. No NaNs in tested paths.

The three timing runs held equal tokenizer/data order, seed42, context1024, AdamW, BF16 and 163,840 tokens. One seed, synthetic fixtures, very short warmup: these are engineering observations, not a model ranking. Missing benchmark columns in leaderboard stay blank.

| Model | Train-step tok/s | Short-run wall tok/s | Peak allocated GiB | Val loss |
|---|---:|---:|---:|---:|
| DenseCompute | 27,587 | 13,980 | 1.476 | 9.81031 |
| SparseV3 | 15,122 | 9,459 | 1.561 | 9.73151 |
| SparseV3Ngram | 15,428 | 9,757 | 1.633 | 9.76573 |

Short-run wall throughput includes final checkpoint/validation, whose fixed cost is poorly amortized over only 20 steps. Training initialization is excluded. The five-minute standalone DenseCompute profiler completed 300.04 seconds / 9,955,328 tokens at 33,180 tok/s, microbatch2, context1024. Sparse30-second profiles measured 9,591 and 10,448 tok/s. These profiler rates are single-microbatch optimizer steps, not the accumulation4 run.

One-minute timer test: 1,605,632 tokens / 59.41 seconds = 27,026 wall tok/s; checkpointed at step 196. Validation was correctly skipped near the timer limit. A brief GPU regression test overlapped the beginning of this timer test, so this rate is a conservative local observation, not an isolated hardware benchmark. A separate continuation restored the timer checkpoint.

## Calculated duration, not a completed training benchmark

| Tokens | Dense timer wall-rate estimate | Dense five-minute profiler estimate |
|---:|---:|---:|
| 20,000,000 | 12.33 min | 10.05 min |
| 50,000,000 | 30.83 min | 25.12 min |
| 100,000,000 | 61.67 min | 50.23 min |

Formula: tokens / measured tokens_per_second. These are extrapolations over a synthetic corpus. Additional initialization, periodic validation, checkpoint costs, thermals, real-corpus lengths and dispatch behavior can change duration. Batch finder recommends larger batches, but the first command keeps microbatch2 for the verified fairness configuration.

## Experimental outcomes and limitations

- Shared-MTP three-step smoke: horizon accuracies were zero on the bounded held-out sequence. Eight-token greedy speculation had accepted length 0.80, exact output parity, and net speedup 0.80x. No MTP speed benefit established; full-prefix reference has no KV cache.
- RouteAhead ten-step synthetic predictor: Top1 next-expert recall 0.208. Prefetch improved simulated stall in 0/1 held-out sequences. No prefetch hypothesis success claimed.
- 4,096 real micro-model routed-token records collected. Atlas fitted first 2,048, cache/packing evaluated next 2,048; 32 cache scenarios. Scaling simulation is explicitly synthetic layer replication/expert remapping, not future-large-model evidence.
- 100 exact-fact tasks / 3,200 deterministic archive-retention checks generated. One-task 1k neural answer-NLL/teacher-forced-accuracy smoke ran for 32 states; prompt truncation recorded. Dictionary retention/oracle retrieval is not model task pass rate. 8k/16k quality remains unmeasured.
- N-gram sample diagnostics include utilization, collision and frequency histograms. Capacity optimality, Top1/Top2 quality tradeoff, two-seed real-code comparisons, meaningful expert specialization and TPU MFU remain unmeasured.
- Optional Strata capture failed (exit 2): Native IQ pack requires --spec T>=2 and --prefill. Its layer-major speculative dumps have no token/attempt position IDs, so the strict single-token adapter cannot establish token alignment without changing runtime. Optional dataset skipped; no installation patched. Supplied Alles-Start.bat was read-only configuration evidence; its interactive Claude CLI was not launched.
- Integration bug found and fixed: CUDA verification under no_grad needed enable_grad for its backward probe. Regression and trace collection now pass. No unresolved core unit-test failure.
- Long pretraining, teacher SFT, page-aware routing, alternative attention, physical asynchronous offload, optimized speculation and secure VM/container agent execution are later lanes. No architecture winner selected.

## First safe manual run

From D:\SparseCoderLab:

```powershell
Set-Location D:\SparseCoderLab
.\.venv\Scripts\python.exe -m tools.train --config configs/dense_compute.yaml --run-dir experiments/phase1_dense_seed42 --target-tokens 5000000 --max-wall-minutes 20
```

Expected throughput-only duration: 5,000,000 / 27025.76 = 185.0 seconds (3.08 minutes). Allow roughly 3–5 minutes including variation/periodic evaluation; hard wall cap is 20 minutes. This uses synthetic development data and is an engineering Phase-1 run, not evidence about real coding quality.

Resume with the same command plus `--resume experiments/phase1_dense_seed42/checkpoints/last.pt`. The target token count remains cumulative. Use at most the default 255 minutes for a later authorized daily run, after matched short research results and current phase-0 gate. No later phase starts automatically.

Evidence: results/tests.xml, test_output.txt, gpu_verification.json, environment.json, parameter_counts.json, profile_*.json, timer_before_resume.json, phase0_gate.json, leaderboard.csv and individual experiments/*/summary.json.

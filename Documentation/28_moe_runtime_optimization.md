# Prompt-1: fix and characterize MoE execution overhead

Case A, with a qualification: runtime is sufficiently close to dense to justify a separately authorized real-data quality experiment, but it remains about 38–40% more costly. This meets the user's potentially acceptable <=1.50 band; it is not the <=1.35 strong or <=1.20 excellent result. No real corpus or 20M/50M/70M/100M run was started.

Historical Phase1A training-step cost was 2.79x dense (Ngram 2.98x), with tiny synthetic loss differences that established no quality winner. Those results remain untouched in Documentation 23–27, the historical leaderboard and snapshots. This prompt begins at commit `0e36357` and only changes execution/measurement infrastructure. Current wall measurement excludes evaluation/checkpoint costs, unlike historical wall rates.

## Measured cause and remaining cost

The focused reference profile ran before any dispatch change, using the actual trained SparseV3 checkpoint, context 1024/microbatch 2/accumulation 4, three warm and three profiled updates. It observed 432 separate expert invocations; last recorded three-layer group snapshots ranged68–247 with mean 170.67 tokens/expert. Width320/intermediate 160 GEMMs at those sizes are small. GPU time attributed to expert GEMM/SwiGLU scopes was only 5.61 ms while their CPU scopes occupied296.29 ms. The evidence points to eager dispatch, indexing/synchronization and many small launches, rather than an intrinsic 2.8x active-compute disadvantage.

| Reference scope | CPU total ms | % of CPU forward scope | Attributed device ms |
|---|---:|---:|---:|
| Router | 10.90 | 0.68 | 0.261 |
| Softmax/Top1 | 9.73 | 0.61 | 0.405 |
| Grouping/indexing | 324.15 | 20.27 | 3.770 |
| Expert GEMMs/SwiGLU | 296.29 | 18.53 | 5.614 |
| Scatter/combine | 144.22 | 9.02 | 3.017 |
| Expert-loop enclosing scope | 790.77 | 49.44 | 12.400 |
| Balance/diagnostics | 68.86 | 4.31 | 1.133 |

The enclosing loop overlaps its children; do not add its percentage to them. Its Python annotation self time was 25.99 ms. The profile's complete CPU forward/backward/clip-optimizer scopes were 1599.31/2200.46/192.29 ms. It recorded42729 CUDA-runtime launch calls,4356 driver launch calls and 504 stream synchronizations; these API counts must not be blindly summed as unique kernel counts. There were 432 `nonzero` calls. CUDA `nonzero` requires host/device synchronization ([official PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nonzero.html)). Total non-annotation CUDA kernel self time was 277.61 ms. GPU-utilization endpoint samples 38%/33% are coarse and do not establish occupancy. Profiler elapsed6.96 s includes teardown overhead, not clean throughput. Autograd device work is not reliably attributed to its enclosing backward CPU annotation; its near-zero annotated device number is not true backward GPU time.

The final grouped profile observes29964 CUDA events versus 49923 reference (event count is not a guaranteed kernel count),25818 runtime launches and 2772 driver launches. CPU forward drops to875.51 ms and backward to1283.61 ms; clip/optimizer remains183.27 ms. Group counting takes10.27 ms and complete grouped dispatch112.66 ms for 36 layer calls. Kernel self time is250.76 ms: only~10% less GPU active work despite a substantial latency gain. This supports the host/launch-overhead diagnosis. Remaining time includes eager backward/launch overhead, unpadded per-expert weight-gradient GEMMs, temporary weight stacking/casts, the shared backbone, large vocabulary log-softmax and SDPA. Largest observed device categories include SDPA backward26.63 ms, log-softmax backward21.10 ms and forward16.36 ms; copy/cast kernels also rank highly. The runtime gain does not imply a corresponding FLOP reduction.

## Execution and correctness

Reference remains default and intact. The optional grouped path uses GPU stable sorting, observed-size padded packing, batched forward/input-gradient GEMMs, unpadded expert weight reductions, exact Top1 multipliers and unchanged balancing. No token drops occur. Registered parameter layout remains identical; old checkpoints load directly, and optimizer handling preserves absent-expert momentum/decay semantics. The first fully padded backward was rejected because tiny gradient rounding differences amplified during trained Top1 continuation. The corrected path passes deterministic FP32/BF16 forward, routing, gradients and five optimizer updates, including the existing checkpoint and AdamW state: all reported final differences are zero on the tested trials. Native reference-versus-reference BF16 continuation is itself nondeterministic. Forty tests and the current-source engineering gate pass. See Documentation 30 for declared tolerances, failed/control trials and exact checkpoint/optimizer requirements.

## Layer speed

The final 96-row matrix covers contexts 256/1024/2048, batches 1/2/4/8 and four distributions. Median forward+backward speedups are3.50x balanced,3.32x moderate,3.13x strong and 3.46x natural. Natural-case range is3.10–3.76x. At context 1024/microbatch 2, reference versus grouped forward is13.250/3.117 ms and forward+backward26.199/7.043 ms:4.25x and 3.72x speedups. All twelve expert groups are nonempty, counts83–245, padded capacity 256 and padding ratio1.5. Strong skew remains faster in every tested case (minimum1.77x), but padded work can exceed10x real tokens. Batch8 is the largest configured candidate tested, not a claim of absolute safe maximum. Layer memory includes the common capture/checkpoint fixture, and short GPU-utilization fields are intentionally blank.

## Matched full-model speed

Rates aggregate two reverse-order repetitions; latency is the median of each repetition's warmed six-call median. Twenty measured updates per distinct loop, identical context 1024/microbatch 2/accumulation 4/BF16/data/policy. Full protocol, scopes and source/data hashes are in Documentation 31 and `results/moe_benchmark_provenance.json`.

| Model | Step tok/s | Wall-loop tok/s | Forward ms | Forward+backward ms | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| DenseCompute | 25273 | 25568 | 30.290 | 73.773 | 1.469 | 1.838 |
| Sparse reference | 11266 | 11275 | 69.078 | 170.036 | 1.566 | 1.713 |
| Sparse grouped | 18361 | 18289 | 40.648 | 96.647 | 1.597 | 1.979 |
| Grouped + current Ngram | 17792 | 17651 | 42.035 | 98.764 | 1.661 | 1.998 |

Grouped improves full-model step throughput1.630x and wall-loop throughput1.622x versus reference. Its cost ratio is1.376x dense for pure steps and 1.398x for wall loops. Current matched reference costs2.243x/2.268x dense; do not mix current numerator with the older30324 tok/s dense denominator. Individual step ranges are24990–25562 dense,11092–11445 reference and 18108–18621 grouped. These are short engineering repetitions, not confidence intervals or cross-seed quality replications.

Ngram adds3.20% pure-step time and 3.62% wall-loop time, plus69141504 allocated bytes (~65.94 MiB). Its gate/table remain intact. These runtime costs do not establish aggregate quality benefit; prior memory-quality uncertainty remains.

## Stored-capacity-matched geometry

All variants initialize freshly with seed42, same data/optimizer, ten updates per separate pure/wall loop and two reverse-order repetitions. Routed weights are exactly 5529600 in each: E*FFN=1920 over three capacity layers. Active compute changes, so the comparison is explicitly not active-compute matched.

| Experts x FFN | Routed params | Stored params | Active estimate | Step tok/s | Wall-loop tok/s | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| 12 x 160 | 5529600 | 21562560 | 16493760 | 18618 | 18541 | 1.675 |
| 6 x 320 | 5529600 | 21556800 | 16948800 | 20370 | 19933 | 1.676 |
| 4 x 480 | 5529600 | 21554880 | 17407680 | 20327 | 20397 | 1.668 |

Six experts wins pure-step speed by only 0.22% over four; four wins wall-loop speed. Treat6/4 as tied at this noise level, both roughly9% faster than twelve despite larger active estimates. This supports a tiny-expert overhead effect but does not select a research architecture. Warm-first to last pure-loop losses decrease10.422->9.937,10.462->9.961 and 10.436->9.951 respectively, finite smoke evidence only. Keep the twelve-expert quality baseline unless a later controlled quality prompt authorizes a variant.

## Decision, reproducibility and archive

Case A is qualified engineering readiness: proceed in the next prompt to verified real data and cumulative20M->50M->70M->100M checkpoints, subject to its own protocol/tokenizer review. No next phase was started. Optional bounded follow-ups can test fusion/compile of shared/eager work and reductions with the same strict gate; custom Triton, changed permanent weight layout and exotic dependencies were unnecessary here and remain untested. Worst-skew padding, a remaining~40% dense cost, default BF16 numerical repeatability, representative routing/locality, real quality, multiple seeds and useful memory benefit remain unresolved.

Prompt-1 goal is `GOALS/Prompt-1.md`. The final frozen second copy is `~\Data-Zip\Prompt-1\Prompt-1.zip`, with external `MANIFEST.json`, `GOAL.md` and verification receipt. Canonical working files remain in place; old measurements, failed trials and large-checkpoint identities are preserved. Follow Documentation 29 for future prompt numbering and verified SHA256 archive convention. The archive is refreshed only after these documents and FINAL_STATUS are finalized.

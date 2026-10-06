# FINAL_STATUS — Prompt-1 MoE runtime gate, 2026-10-05

**Case A, qualified:** grouped SparseV3 now costs 1.38x dense for training steps and 1.40x for the short wall loop. Correctness passes; this is potentially acceptable under the requested engineering bands. Recommend the next, separately authorized real-data cumulative 20M -> 50M -> 70M -> 100M experiment. No corpus acquisition or long training started.

| Matched RTX5070 model | Training-step tok/s | Wall-loop tok/s | Forward ms | Forward+backward ms | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| DenseCompute | 25273 | 25568 | 30.290 | 73.773 | 1.469 |
| Sparse reference | 11266 | 11275 | 69.078 | 170.036 | 1.566 |
| Sparse grouped | 18361 | 18289 | 40.648 | 96.647 | 1.597 |
| Sparse grouped + Ngram | 17792 | 17651 | 42.035 | 98.764 | 1.661 |

Two reverse-order repetitions; context 1024/microbatch 2/accumulation 4/BF16, same frozen tokenizer/shards and step policy. Each twenty-step pure/wall loop is a bounded speed experiment. Rate is total tokens / total seconds; forward latencies summarize warmed six-call medians. Wall includes batching/logging but excludes initialization, evaluation and checkpoint I/O. Its rate can slightly exceed the separate pure-step loop because of timing variation. These wall values do not replace historical Phase1A wall rates.

1. **Original slowdown:** eager per-expert dispatch, small GEMMs, repeated indexing/synchronization and launch overhead. Before optimization the profile observed 432 expert forwards, 432 nonzero calls and 42729 runtime launch calls in three updates; expert GEMM scopes used only 5.61 ms attributed device time versus 296.29 ms CPU scope. This is measured execution evidence, not a theoretical FLOP claim.
2. **Backend speedup:** final natural-routing MoE forward+backward median 3.46x over the layer matrix; 3.72x at context 1024/batch 2. Full-model step/wall speedups 1.630x/1.622x against current reference. Rejected padded-backward timings are preserved but excluded from these claims.
3. **Dense rate:** 25273 step, 25568 short wall-loop tok/s.
4. **Sparse reference rate:** 11266 step, 11275 wall-loop tok/s.
5. **Sparse grouped rate:** 18361 step, 18289 wall-loop tok/s.
6. **Sparse/dense cost:** 1.376x step, 1.398x wall. Historical Phase1A's 2.79x step penalty remains its original observation; current matched reference ratio is 2.243x.
7. **Ngram increment:** +3.20% step time, +3.62% wall time, +69141504 allocated bytes (~65.94 MiB). Existing memory design remains intact; quality benefit is still unproved.
8. **Geometry:** at exactly 5529600 routed stored weights, 12x160 gets 18618 step tok/s, 6x320 gets 20370, 4x480 gets 20327. Six/four are effectively tied: 6 wins pure steps by 0.22%, 4 wins wall. Active estimates differ (16.494M/16.949M/17.408M); all fresh short smokes have finite falling loss. No quality winner or automatic geometry switch.
9. **Parity:** FP32 and RTX BF16 forward/routing/auxiliary/intermediate outputs, all gradients and five optimizer updates pass, including old trained model and AdamW state. Final deterministic tested differences are all zero; checkpoint layout/order and empty-expert optimizer semantics are retained. Native BF16 reference itself is nondeterministic, so universal bitwise continuation is not promised. Forty regressions and the current-source Phase0 engineering gate pass.
10. **Next phase justified:** yes as a separately planned quality experiment under the potentially acceptable <=1.50 runtime band. This is weaker than the <=1.35 strong band. No next phase started.
11. **Unresolved:** remaining eager backward/launch/copy/stacking overhead; skew padding; ~40% dense cost; native numerical repeatability; real-data/coding quality, tokenizer suitability, multiple seeds, routing specialization/locality, useful memory benefit and larger-model scaling. Compile/fusion/custom kernels were unnecessary for this prompt and remain untested.

The optional grouped implementation preserves reference Top1 multipliers, auxiliary balance and no-drop routing. Existing reference remains default. Temporary stacked weights preserve all registered parameter/state_dict names and existing checkpoints. `make_optimizer` selects GroupedAdamW to preserve None gradients and skip updates for experts absent over an accumulation interval. Original Phase1A configs, data, checkpoints and quality leaderboard are unchanged; historical context is in Documentation23–27 and `results/history/Prompt-1/`.

The first fully padded backward failed trained continuation parity and was rejected. A reference-versus-reference control also failed under native nondeterministic BF16; deterministic control resolved that issue. The final unpadded weight-gradient reduction passes the unchanged declared tolerances. All failed trials remain visible in history; Documentation30 explains them and the existing circular-import repair.

Canonical goal: `GOALS/Prompt-1.md`. Main report: `Documentation/28_moe_runtime_optimization.md`; correctness/protocol/archive details are separate files29–31. Raw final measurements and source/data/CSV hashes live under `results/moe_*`. Historical quality losses are not reinterpreted or mixed into these speed-only measurements.

Frozen second copy: **~\Data-Zip\Prompt-1\Prompt-1.zip**, with `MANIFEST.json`, `GOAL.md` and `VERIFICATION.json`. After final documentation the archive is opened and CRC/SHA256/project-relative paths/current canonical bytes/goal mirror are verified by `tools/archive_prompt.py`; `results/prompt-1_archive_verification.json` is the machine-readable receipt. Large checkpoints are recorded by path/size/SHA256/origin rather than copied. Canonical working files remain in D:\SparseCoderLab. Future prompts follow Documentation29's next-unused-number convention.

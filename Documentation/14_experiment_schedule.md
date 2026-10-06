# Successive halving and gates

| Phase | Scope | Promotion condition |
|---|---|---|
| 0 | Unit/integration tests, tiny overfit, NaN/gradients, exact restart, counts, five-minute profiler | All core tests pass |
| 1 | DenseCompute / SparseV3 / SparseV3+memory, 5–20M tokens or small wall cap | Stable, measured quality/compute tradeoff |
| 2 | Survivors, 30–50M tokens if measured runtime permits | Compare equal tokens, estimated FLOPs and wall clock |
| 3 | Best sparse and strongest dense, daily 255-minute cap | At least two seeds where compute permits |
| 4 | Shared MTP3 then optional independent steps | Net acceptance and throughput justify cost |
| 5 | Context/compaction/retrieval | Exact-fact retention and model use measured separately |
| 6 | Routing traces, atlas, caches and RouteAhead | Held-out trace I/O improvement |

No phases automatically chain. Commands exceeding 30 minutes require a current-source `results/phase0_gate.json` from `python -m tools.phase0`. Source/config/test/lock changes invalidate that hash. This gate does not schedule training or authorize a long job. The initial task stops after short checks and reports a command for manual launch.

One seed is sufficient for engineering smoke and initial elimination; final short comparisons should use two seeds. Predeclare checkpoint/token comparison points. Kill unstable variants explicitly and retain their logs. A sparse model is not 'better' without naming equal tokens, active compute, stored size or wall-clock basis.

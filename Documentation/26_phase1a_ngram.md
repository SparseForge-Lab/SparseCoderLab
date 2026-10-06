# Phase 1A N-gram contribution

The gate and gradients are active, but there is no established aggregate validation benefit. These are different measurements.

Gate mean on the logged last microbatch rises from 0.119141 at step 10 to 0.625185 at step 610. Last gate range is 0.155273–0.980469. All four banks have nonzero table gradients at all 61 logged steps. These are sampled update observations, not exact lifetime row frequencies. AdamW also applies dense weight decay and momentum updates; nonzero gradient rows must not be equated with every changed parameter row.

| Bank | Final sampled post-clip gradient norm | Gradient rows in accumulated step | Utilized buckets in last microbatch | Distinct ngrams | Collisions | Collision fraction |
|---|---:|---:|---:|---:|---:|---:|
| 0 / order 2 | 0.002608 | 5364 | 1587 / 131072 | 1606 | 19 | 1.1831% |
| 1 / order 2 | 0.002380 | 5365 | 1587 / 131072 | 1606 | 19 | 1.1831% |
| 2 / order 3 | 0.002543 | 6290 | 1743 / 131072 | 1753 | 10 | 0.5705% |
| 3 / order 3 | 0.002828 | 6290 | 1743 / 131072 | 1753 | 10 | 0.5705% |

Collision counts/distributions are calculated from the bounded 2,048-token last microbatch. They are not whole-corpus or lifetime utilization. Full frequency/collision histograms over the sampled checkpoints are retained in raw metrics.jsonl. The two banks of a given order have identical counts in this final sample; this observation alone does not establish independent hashing.

| Evaluation | Enabled | Residual zeroed | Ablated minus enabled |
|---|---:|---:|---:|
| val_loss | 4.34909388 | 4.34914789 | +0.00005400 |
| code_val_loss | 3.98227521 | 3.99025000 | +0.00797478 |
| general_val_loss | 5.07993221 | 5.07987534 | -0.00005687 |
| bits_per_token | 6.27441618 | 6.27449409 | +0.00007791 |

Both evaluations use exactly the same trained 5,005,312-token checkpoint and deterministic samples. No parameters were changed or retrained. The evaluation-only flag returns the input unchanged at the memory injection point, so the residual contribution is exactly zero; training with this flag raises an error. Unit tests verify zeroing and restoration.

The mixed NLL effect is only +0.000054 when ablated. The code subset changes by +0.007975 and general by -0.000057. This weak subset-specific effect is measured on repetitive fixtures; no across-seed reproduction or real-code benefit is established. A learned gate of 0.625 is insufficient evidence by itself. Half/current capacity training was not launched because the overall slowdown review stop takes precedence over further experiments. No large-model memory-capacity optimum is inferred.

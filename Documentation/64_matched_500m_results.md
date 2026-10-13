# Matched repository-data comparison at 500M tokens

Dense, Sparse + 10M Ngram and Sparse + 25M Ngram completed training and matched evaluation at exactly **499,998,720 total tokens** (61,035 total updates). Each continued from 250,003,456 tokens, adding 249,995,264 tokens in 30,517 updates. Plain Sparse was intentionally excluded from this continuation; its matched 250M result remains historical evidence.

Final full-state receipts, model-only identities and evaluation results agree. A CPU-only reconciliation on 2026-10-13 rehashed all six final payloads and inspected full-state model/optimizer finiteness and token counters. Existing loader-verification records additionally bind scheduler, cursor, RNG state and model-only equality. This update did not rerun training, generation or functional evaluation.

The [aggregate evidence record](../results/prompt5_500M_v1/reconciled_500M_evidence.json) identifies the primary inputs by SHA-256. The [all-arm gate manifest](../results/prompt5_500M_v1/final_gate/matched_gate_checkpoints.json) is authoritative for matched coverage. `training_complete.json` records the last selected +10M invocation only, not all three arms. Likewise, each `provenance.json` is a historical resume-load snapshot; use `summary_500M.json` and `checkpoint_verification.json` for final counters.

## Held-out negative log-likelihood

| Model | Mixed | Code | Technical | General |
|---|---:|---:|---:|---:|
| Dense | 2.688166600 | 2.099091223 | 5.983665758 | 2.732021402 |
| Sparse + 10M Ngram | 2.659873122 | 2.073929499 | 5.950040485 | 2.683864355 |
| Sparse + 25M Ngram | 2.631813498 | 2.059776317 | 5.873659584 | 2.659208996 |

The exact best mixed NLL is **2.631813498392804** for +25M. Mixed NLL uses 128 frozen packed batches (1,048,064 predictions); category NLL uses independently hash-seeded eligible document windows. Category averages cannot reconstruct mixed NLL. All three reports share the frozen evaluation-index hash `841568bb647907508b9e2bed8e09de90ef65c13c56103818cf9ee79a6d62bbf1`.

## Three distinct executable metrics

| Model | Matched executable | Strict target function | Strict whole output |
|---|---:|---:|---:|
| Dense | 0/175 | 0/175 | 0/175 |
| Sparse + 10M Ngram | 4/175 | 3/175 | 5/175 |
| Sparse + 25M Ngram | 4/175 | 9/175 | 3/175 |

The [matched executable report](../results/prompt5_500M_v1/final_gate/functional_500M_r1.json) executes prompt plus completion and checks the frozen function cases. Its saved matrix uses batch size 4, top-p 0.95, top-k 40, a 128-token cap and seed 42 across seven temperatures. The report records the exact legacy guest-wrapper identity used by the canonical baseline; the current wrapper differs, so reproducing this historical score requires that pinned wrapper from commit `221553b`. No scoring policy was changed during reconciliation.

The [target and whole-output report](../results/prompt5_500M_v1/final_matched_functional.json) instead scores the separate milestone generation matrix (top-p disabled, top-k 40, the same cap, seed and temperatures). Its [scorer](../forgebuilder/functional_eval.py) extracts the earliest complete requested function, executes it and checks explicit constraints such as loop use and returning a fresh list. The whole-output rule executes the complete normalized candidate and checks function behavior and module output. Markdown fences are normalized; this is not a byte-for-byte raw-text acceptance rule. These definitions are not nested: +10M has five whole-output passes but only three target passes. Neither report should be merged with the other.

Each denominator is 25 prompts at seven temperatures, not 175 independent tasks. These single-seed finite-case scores are not a conventional pass@k estimate or evidence of general software-engineering capability.

## Stopping and scope

| Model | Inferred natural stops | Cap reached | Continuation after target | Repetition after target | Correct target later ruined |
|---|---:|---:|---:|---:|---:|
| Dense | 35 | 140 | 83 | 31 | 2 |
| Sparse + 10M Ngram | 58 | 117 | 85 | 29 | 0 |
| Sparse + 25M Ngram | 22 | 153 | 106 | 34 | 8 |

Counts are per 175 milestone samples. Natural stopping is inferred from generation length below the cap because EOS identity was not saved. Tail lengths are approximate re-encodings. Reaching the cap is reported separately and does not automatically fail an executable candidate. +25M frequently continues after a useful function; nine target passes versus three whole-output passes expose a remaining stopping/scope weakness.

## Intermediate +10M diagnostics

| Total tokens | Parseable prompt + output | Repetition-loop flag |
|---:|---:|---:|
| 374,996,992 | 69/175 | 67/175 |
| 466,944,000 | 52/175 | 81/175 |
| 499,998,720 | 68/175 | 62/175 |

The 374,996,992 → 466,944,000 → 499,998,720 trajectory is non-monotonic on these syntax/repetition indicators. It motivates an additional midpoint in future comparisons, but does not establish a functional-correctness or NLL degradation at 466,944,000 tokens; those intermediate executable measurements were not made.

## Memory, routing and efficiency evidence

| Variant | Residual-off mixed NLL increase | Code | Technical | General |
|---|---:|---:|---:|---:|
| Sparse + 10M Ngram | 0.024312744 | 0.051291037 | 0.020203938 | 0.035003616 |
| Sparse + 25M Ngram | 0.051359977 | 0.082540737 | 0.043813281 | 0.055555541 |

Disabling the memory residual increases all four losses within each trained checkpoint. This demonstrates checkpoint dependence on the residual, not superior knowledge storage or a causal explanation of cross-architecture differences.

Both sparse reports contain 56 language/domain-layer routing records with zero unused experts per record. Maximum expert share ranges from 9.51% to 26.49% for +10M and 10.17% to 29.38% for +25M. Stratum gate means range from 0.5685 to 0.7687 and 0.6145 to 0.7920 respectively. Sample collision fractions range from 0 to 0.006873 and 0 to 0.005556. These are bounded-sample lookup/collision statistics, not whole-corpus utilization or proof of useful semantic specialization.

Historical training logs and [generation-efficiency groups](../results/prompt5_500M_v1/final_gate/generation_efficiency_500M.json) retain timing and memory evidence. Logged training peaks were approximately 1.44/1.87/2.03 GiB allocated for Dense/+10M/+25M. The final NLL passes recorded 37.31/58.74/58.53 seconds, excluding separate residual-off passes. Desktop/game activity and different sequence lengths affect these timings; they are not controlled architecture speed claims.

## Checkpoint identities

| Model | Full-state SHA-256 | Model-only SHA-256 |
|---|---|---|
| Dense | `578cdf1b4bdf7bc81e0f95d5817f85741fb83daac598b2c9c95c8ca646240548` | `81f257e0e1e0a80c309ef26503d547eea7705f9b467cd2b42949fd2223b6d05d` |
| Sparse + 10M Ngram | `8ef8d7e1ca7bcba14e3a2bee8e7438ded774dcd629f04ba91ae7dda00d3c47c4` | `645d75e3d77ea48ea604cffa4f68f36d67f978226941da5ccf4cf76f36e79057` |
| Sparse + 25M Ngram | `953ebf1c89c05f0591404709d0f7070c49f0998f6d3a0943ceb7e8c854b911ce` | `6ba76650bfacba91e02df7c9584f35f2bf20ca461976a28f7627d4742b12e81b` |

Payload paths and byte sizes are recorded in the linked aggregate and per-model `summary_500M.json` records. Large checkpoints remain canonical and outside normal Git and ZIP payloads.

## Interpretation and planned comparison

+25M currently has the strongest strict target-function result and lowest NLL; +10M remains a useful control and has more whole-output passes under that particular rule. Dense is weakest on these functional measures. The small sample, one training seed, differing score definitions and stopping failures do not establish a final architectural winner, general MoE superiority, or an Ngram causal mechanism.

The next controlled comparison retains both Ngram variants, resuming their exact 500M full states on the same Dataset V2 stream under matched conditions. The mandatory midpoint is **625,000,448 tokens / 76,294 total updates**, followed by **750,002,176 tokens / 91,553 total updates**. Relative to 61,035 starting updates these add 15,259 and 30,518 updates. The midpoint is exactly halfway between the aligned endpoints. Dataset V2 training and both future gates are **planned / not started**, and require separate launch authorization.

At both gates, retain NLL by stratum, all three executable definitions, length/stop/cap/repetition measures, router and memory diagnostics, residual-off ablation, training/inference efficiency, and checkpoint hashes/provenance. The [disabled comparison plan](../configs/dataset_v2/comparison_plan.json) and [infrastructure report](dataset_v2_and_forgebuilder.md) describe the preparation state. No new data-generation or training run was launched for this reconciliation.

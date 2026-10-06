# Matched20M evidence — partial Prompt-2 comparison

All three seed42 models completed20,004,864 cumulative predicted tokens /2,442 updates with the same frozen research corpus, tokenizer, evaluation, token order and100M schedule. Lower NLL is better. The50M matched review and70M/100M comparisons are still required.

| Model | Mixed NLL | Code NLL | General NLL | Technical NLL |
|---|---:|---:|---:|---:|
| dense | 4.302957 | 3.757303 | 5.917018 | 4.588342 |
| sparse | 4.464748 | 3.916070 | 5.969997 | 4.714492 |
| memory | 4.171885 | 3.623517 | 5.846257 | 4.489081 |

Ngram-Sparse mixed/code deltas are−0.292863/−0.292553 nats/token; Ngram-Dense−0.131072/−0.133786. The mixed perplexity ratio versus Sparse is0.7461 (calculated). Sparse alone remains behind Dense at20M, whereas its already measured50M mixed/code gap changed sign. One matched Ngram point cannot establish its curve shape or a final architecture winner.

## Same-checkpoint memory ablation

| Category | Zero residual minus normal NLL |
|---|---:|
| Mixed | +0.059097 |
| Code | +0.094661 |
| General | +0.035774 |
| Technical | +0.109459 |

The positive penalty measures useful residual contribution within this trained checkpoint. It does not attribute the entire cross-model gap to inference-time lookup: memory also changes optimization and initialization trajectories. This first real-data penalty is larger than the historical tiny5M synthetic effect, but different corpora/training endpoints prevent a causal attribution solely to more tokens. Gates average0.96884 code,0.95302 general,0.96729 technical and0.97069 context; a near-open gate alone is not usefulness evidence.

## Routing and sampling

| Model | Layer | Load CV | Max share | Unused experts | Category-expert MI bits |
|---|---:|---:|---:|---:|
| sparse | 2 | 0.1801 | 0.1209 | 0 | 0.0305 |
| sparse | 5 | 0.1890 | 0.1126 | 0 | 0.0497 |
| sparse | 8 | 0.1536 | 0.1127 | 0 | 0.0624 |
| memory | 2 | 0.2133 | 0.1157 | 0 | 0.0365 |
| memory | 5 | 0.1549 | 0.1108 | 0 | 0.0529 |
| memory | 8 | 0.1745 | 0.1167 | 0 | 0.0388 |

No expert is unused in pooled held-out routes. Modest category associations are descriptive, not proof of useful semantic specialization. Rust/SQL standalone strata remain insufficient.

Ngram-Sparse code paired repository-cluster95% interval: [−0.309336,−0.275109]; general [−0.138357,−0.108056]; technical [−0.278215,−0.179469]. These intervals quantify fixed document-sample variation, not training-seed variance. The all-document interval combines the diagnostic category documents and must not be presented as the separate packed mixed-validation interval.

All three models passed0/6 bounded arithmetic-code probes at20M. Lower prediction loss does not establish useful coding/chat capability.

## Checkpoints and continuation

Ngram checkpoint SHA25667920b5f5da6b439154d5751e41b6f94016a465360e89a362f5c562688175190. Permanent ledger and20M release snapshot contain checkpoint paths/sizes/hash, full metrics, source/config/data/tokenizer identities and architecture. Ngram continues cumulatively20M→50M; no new seed, LR schedule or corpus is introduced.

Ngram20M measured22,217 step tok/s and20,041 wall tok/s, with1,894,864,384 peak allocated bytes. Cumulative speed includes user-authorized Roblox/desktop concurrency; Dense wall retains the documented reporting-recovery lower-bound caveat. Prompt-1 remains the controlled speed study.

Sources: immutable checkpoint summaries; results/research_v1/review_evidence.json, memory_20004864_ablation.json, clustered_document_intervals.json, ngram_gate_summary.json and resume_state.json; results/releases/20M/.

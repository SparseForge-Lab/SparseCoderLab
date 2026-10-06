# 20M research snapshot

Status: matched_complete. One seed, fixed research tokenizer/corpus. Partial snapshots do not establish matched architecture conclusions.

| Model | Actual tokens | Mixed NLL | Code NLL | General NLL | Step tok/s |
|---|---:|---:|---:|---:|---:|
| ResearchV1_DenseCompute | 20004864 | 4.302957 | 3.757303 | 5.917018 | 30090 |
| ResearchV1_SparseV3_Grouped | 20004864 | 4.464748 | 3.916070 | 5.969997 | 22528 |
| ResearchV1_SparseV3_GroupedNgram | 20004864 | 4.171885 | 3.623517 | 5.846257 | 22217 |

Canonical checkpoint paths/SHA256/size/architecture are in summary.json; full evaluation and bounded routes in metrics.json. Rust/SQL standalone claims are insufficient. Dense wall is a documented lower bound. No SOTA, useful coding-agent or large-model-transfer claim.

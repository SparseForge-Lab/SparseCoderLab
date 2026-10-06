# 50M research snapshot

Status: partial. One seed, fixed research tokenizer/corpus. Partial snapshots do not establish matched architecture conclusions.

| Model | Actual tokens | Mixed NLL | Code NLL | General NLL | Step tok/s |
|---|---:|---:|---:|---:|---:|
| ResearchV1_DenseCompute | 50003968 | 3.288592 | 2.778863 | 5.113289 | 31892 |
| ResearchV1_SparseV3_Grouped | 50003968 | 3.276931 | 2.756853 | 5.109978 | 23367 |

Canonical checkpoint paths/SHA256/size/architecture are in summary.json; full evaluation and bounded routes in metrics.json. Rust/SQL standalone claims are insufficient. Dense wall is a documented lower bound. No SOTA, useful coding-agent or large-model-transfer claim.

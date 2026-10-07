# 50M research snapshot

Status: matched_complete. One seed, fixed research tokenizer/corpus. Partial snapshots do not establish matched architecture conclusions.

| Model | Actual tokens | Mixed NLL | Code NLL | General NLL | Step tok/s |
|---|---:|---:|---:|---:|---:|
| ResearchV1_DenseCompute | 50003968 | 3.288592 | 2.778863 | 5.113289 | 31892 |
| ResearchV1_SparseV3_Grouped | 50003968 | 3.276931 | 2.756853 | 5.109978 | 23367 |
| ResearchV1_SparseV3_GroupedNgram | 50003968 | 3.263211 | 2.743938 | 5.074183 | 21966 |

Checkpoint SHA256, size, and architecture are recorded in summary.json; full evaluation and bounded routes are in metrics.json. Checkpoint payloads remain local. Rust/SQL standalone claims are insufficient. Dense wall is a documented lower bound. No SOTA, useful coding-agent or large-model-transfer claim.

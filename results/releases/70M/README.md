# 70M research snapshot

Status: matched_complete. One seed, fixed research tokenizer/corpus. Partial snapshots do not establish matched architecture conclusions.

| Model | Actual tokens | Mixed NLL | Code NLL | General NLL | Step tok/s |
|---|---:|---:|---:|---:|---:|
| ResearchV1_DenseCompute | 70000640 | 3.106532 | 2.608955 | 4.937882 | 31761 |
| ResearchV1_SparseV3_Grouped | 70000640 | 3.083776 | 2.579422 | 4.902308 | 23322 |
| ResearchV1_SparseV3_GroupedNgram | 70000640 | 3.078860 | 2.574937 | 4.879346 | 22024 |

Checkpoint SHA256, size, and architecture are recorded in summary.json; full evaluation and bounded routes are in metrics.json. Checkpoint payloads remain local. Rust/SQL standalone claims are insufficient. Dense wall is a documented lower bound. No SOTA, useful coding-agent or large-model-transfer claim.

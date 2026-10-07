# 100M research snapshot

Status: matched_complete. One seed, fixed research tokenizer/corpus. Partial snapshots do not establish matched architecture conclusions.

| Model | Actual tokens | Mixed NLL | Code NLL | General NLL | Step tok/s |
|---|---:|---:|---:|---:|---:|
| ResearchV1_DenseCompute | 100007936 | 3.010259 | 2.517283 | 4.804755 | 31376 |
| ResearchV1_SparseV3_Grouped | 100007936 | 2.981258 | 2.481819 | 4.775507 | 22673 |
| ResearchV1_SparseV3_GroupedNgram | 100007936 | 2.974749 | 2.476290 | 4.750409 | 21641 |

Checkpoint SHA256, size, and architecture are recorded in summary.json; full evaluation and bounded routes are in metrics.json. Checkpoint payloads remain local. Rust/SQL standalone claims are insufficient. Dense wall is a documented lower bound. No SOTA, useful coding-agent or large-model-transfer claim.

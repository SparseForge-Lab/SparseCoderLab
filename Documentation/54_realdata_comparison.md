# Real-data architecture comparison

This study compares Dense, Sparse, and Sparse + Ngram models on a fixed real-document corpus. It measures language-modeling loss and routing behavior; it does not test a complete coding agent.

## Setup

All primary models used the same tokenizer, corpus, data order, context length, optimizer policy, and seed. Each reached 100,007,936 training tokens. The selected Sparse/Ngram replication changed the model-training seed to 1337 while keeping the data and evaluation order fixed at seed 42. Full configurations and source identities are recorded in the result manifests.

## Results

| Seed / model | Mixed NLL | Code NLL | General NLL | Technical NLL |
|---|---:|---:|---:|---:|
| 42 Dense | 3.010259 | 2.517283 | 4.804755 | 3.423348 |
| 42 Sparse | 2.981258 | 2.481819 | 4.775507 | 3.397397 |
| 42 Sparse + Ngram | 2.974749 | 2.476290 | 4.750409 | 3.373976 |

The Sparse-minus-Dense mixed/code gap moved from +0.161791/+0.158768 at 20M tokens to -0.029001/-0.035465 at 100M. Ngram's 100M gain over Sparse at seed 42 was smaller: -0.006509 mixed and -0.005529 code nats. At seed 1337, Ngram also outperformed Sparse at 50M, 70M, and 100M, while the 20M ordering reversed. This supports a late-stage gain on the tested data, not a population-level estimate.

Turning off Ngram residuals worsened 100M NLL by 0.092697 mixed, 0.154881 code, 0.089014 general, and 0.155034 technical nats. This shows that the trained Ngram model uses its memory; it does not isolate the causal value of adding memory versus training a different architecture.

All 12 experts received traffic in evaluated sparse layers. Category mutual information remained low (about 0.026–0.040 bits at 100M for seed 42), so useful semantic specialization is not established. All 72 primary arithmetic-code probe attempts failed (0/6 per model at each of four endpoints).

## Interpretation

The fixed-corpus loss curves favor Sparse over Dense after 20M tokens and show a modest late Ngram improvement in both tested seeds. The study's final classification is mixed because the task probes failed, routing evidence is descriptive, and the selected two-seed comparison cannot establish general seed robustness. The measured models also use only a 1,024-token context and do not test instruction tuning, tools, or executable repository repair.

## Reproduction and evidence

Release summaries and evaluations are in `results/releases/`. Detailed curves, router diagnostics, integrity receipts, and seed comparisons are in `results/research_v1/` and `results/replication_seed1337/`. Dataset and checkpoint payloads remain outside ordinary Git; the manifests identify their hashes and sizes. See [the documentation index](README.md) and [model card](../MODEL_CARD.md).

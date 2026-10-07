# Prompt-3 milestone results

This page records matched Prompt-3 checkpoints. All four variants use seed 42 and the same frozen tokenizer, training corpus/order, context length, optimizer policy, and evaluation suite. Prompt-2 remains classified as MIXED; Prompt-3 is a separately tracked architecture study and does not change that result.

## 100M comparison

| Model | Stored / active parameters | Mixed NLL | Code NLL | General NLL | Technical NLL | Heuristic parseable / loop | Train tok/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 54.02M / 54.02M | 2.7500 | 2.2693 | 4.5000 | 3.1485 | 21.7% / 65.1% | 41.8k | 3.26 GiB |
| Sparse | 76.06M / 54.44M | 2.7443 | 2.2501 | 4.4586 | 3.1151 | 19.4% / 72.6% | 40.4k | 8.28 GiB |
| Sparse + 10M Ngram | 86.57M / 54.45M | 2.7294 | 2.2467 | 4.4493 | 3.1160 | 38.3% / 59.4% | 37.5k | 8.47 GiB |
| Sparse + 25M Ngram | 101.25M / 54.45M | 2.7187 | 2.2349 | 4.4426 | 3.1022 | 22.9% / 69.7% | 33.4k | 8.69 GiB |

The generation figures are screening heuristics over 175 continuations per variant, not executable functional scores. “Parseable” means the generated continuation plus its prompt passed the recorded Python parse check; “loop” is the benchmark's repetition heuristic. These results favor the larger Ngram model on held-out NLL, while Sparse + 10M had the strongest heuristic generation screen. Neither is sufficient alone to select a winner.

## 250M comparison

| Model | Tokens | Mixed NLL | Code NLL | General NLL | Technical NLL | Functional coding | Parseable / loop | Train tok/s | Peak VRAM | Promotion |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---|
| Dense | 250,003,456 | 2.4840 | 2.0029 | 4.2245 | 2.8501 | Pending matched benchmark | Pending | 40.0k | 7.28 GiB | CONTINUE provisionally; only 250M candidate evaluated so far |

The 250M same-stage comparison is incomplete. The remaining variants are being evaluated at the same token gate. The 250M evaluation is a single-seed internal proxy, not a public benchmark. No candidate is dropped based on this row alone.

## Milestone evidence

Canonical scaling rows and checkpoint digests are in [`../results/prompt3/milestone_history.csv`](../results/prompt3/milestone_history.csv). Full evaluation reports are retained with each experiment under `experiments/prompt3_v1/<variant>/summary_<gate>.json`; weight and resume checkpoints remain local and are not part of the public repository. Config, corpus, and tokenizer identity are recorded in those summaries. The frozen corpus manifest SHA-256 is `501361addc446c234c765d3634e3401446f0b4a7c65217005b5dcd6aa0b7123e` and the tokenizer SHA-256 is `832daf453fc64e28f50df00ac2e458718576bb26e3f16e737f18b120ae4e309a`.

Functional coding tests, router-health comparisons, and the 500M gate are pending. Promotion at 250M will consider the matched same-stage results together; only survivors proceed to 500M, where one architecture will be selected.

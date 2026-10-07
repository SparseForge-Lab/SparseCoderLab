# 75M model comparison

I tested four versions of a roughly 75M-parameter model: Dense, Sparse, Sparse + 10M Ngram, and Sparse + 25M Ngram. All four reached 100M training tokens with the same seed, tokenizer, corpus, data order, context length, and evaluation setup. The earlier study was mixed, and these results do not change that conclusion.

## Results at 100M tokens

| Model | Stored / active parameters | Mixed NLL | Code NLL | General NLL | Technical NLL | Parseable / loop | Train tok/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 54.02M / 54.02M | 2.7500 | 2.2693 | 4.5000 | 3.1485 | 21.7% / 65.1% | 41.8k | 3.26 GiB |
| Sparse | 76.06M / 54.44M | 2.7443 | 2.2501 | 4.4586 | 3.1151 | 19.4% / 72.6% | 40.4k | 8.28 GiB |
| Sparse + 10M Ngram | 86.57M / 54.45M | 2.7294 | 2.2467 | 4.4493 | 3.1160 | 38.3% / 59.4% | 37.5k | 8.47 GiB |
| Sparse + 25M Ngram | 101.25M / 54.45M | 2.7187 | 2.2349 | 4.4426 | 3.1022 | 22.9% / 69.7% | 33.4k | 8.69 GiB |

The model with 25M of Ngram memory had the lowest held-out losses. In a quick screen of 175 generated continuations per model, the 10M version had the highest Python parse rate and the fewest obvious repetition loops. This was a syntax and repetition screen only: it did not test whether the code worked. The raw outputs and exact prompts remain local; the aggregate counts are in [generation_100M_screen.json](../results/prompt3/generation_100M_screen.json).

## Training setup

The models share a 32,768-token vocabulary, 16 layers, width 512, eight query heads and two key/value heads. The sparse models use grouped Top-1 experts in layers 4, 8, 12 and 16, with 12 experts per routed layer. Their stored parameter count includes all experts; the active estimate counts only the selected expert per token. The memory versions add four hashed tables with 8 dimensions: 10,485,760 entries for the 10M model and 25,165,824 for the 25M model.

Training used BF16 on an RTX 5070 with microbatch 8, accumulation 1, and 1,024-token sequences (8,192 tokens per optimizer update). The shared schedule used a 0.0003 peak learning rate, 2,441 warmup updates and a 122,071-update cosine schedule. At 100M tokens, the corpus contained 210,012,790 training tokens and 8,393,991 validation tokens, mixed as 55% code, 30% general text, 10% technical material and 5% repository context. The corpus-manifest SHA-256 is `501361addc446c234c765d3634e3401446f0b4a7c65217005b5dcd6aa0b7123e`; the tokenizer SHA-256 is `832daf453fc64e28f50df00ac2e458718576bb26e3f16e737f18b120ae4e309a`.

## Later partial runs

The original data and training setup was stopped before all four models reached 250M. These checkpoints are useful exploratory evidence, but they are not a matched 250M comparison.

| Model | Exact tokens | Mixed NLL | Code NLL | General NLL | Technical NLL | Train tok/s | Peak VRAM | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Dense | 250,003,456 | 2.4840 | 2.0029 | 4.2245 | 2.8501 | 39,995 | 7.28 GiB | `5ea3f085…91905fb` |
| Sparse | 180,224,000 | Not evaluated at this stop | Not evaluated at this stop | Not evaluated at this stop | Not evaluated at this stop | 37,277 | 7.90 GiB | `384740ca…411fa860` |

The two Ngram runs remained at 100M. The Dense result has a full evaluation report; Sparse was stopped at its last saved 22,000-step checkpoint and did not receive a separate evaluation at 180.224M. Do not compare these two rows as if they were measured at the same training exposure, and do not treat either as the architecture selection result.

## Transition point

The four 100M checkpoints are the only common comparison point and are now the canonical starting weights for the next data phase. The later partial checkpoints are retained as historical evidence. The next phase changes the data recipe substantially, so it will continue from the matching 100M checkpoint for each model rather than mixing different amounts of prototype-data training into the comparison.

The [checkpoint manifest](../results/prompt3/transition_checkpoint_manifest.json) records paths, sizes, SHA-256 hashes, token counts, parameter counts, configuration identities and resume-state contents. Large checkpoints remain outside Git. No winner has been selected. The main open questions are whether the larger Ngram table improves with more repository data, whether Sparse routing develops useful specialization, and whether any model can solve executable coding tasks rather than merely produce parseable text.

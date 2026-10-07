# 75M model comparison

I am comparing four versions of a roughly 75M-parameter model: Dense, Sparse, Sparse + 10M Ngram, and Sparse + 25M Ngram. They use the same seed (42), tokenizer, training data and order, context length, and evaluation setup. The earlier smaller-model comparison was mixed. This is a separate experiment and does not change that result.

## Results at 100M tokens

| Model | Stored / active parameters | Mixed NLL | Code NLL | General NLL | Technical NLL | Heuristic parseable / loop | Train tok/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 54.02M / 54.02M | 2.7500 | 2.2693 | 4.5000 | 3.1485 | 21.7% / 65.1% | 41.8k | 3.26 GiB |
| Sparse | 76.06M / 54.44M | 2.7443 | 2.2501 | 4.4586 | 3.1151 | 19.4% / 72.6% | 40.4k | 8.28 GiB |
| Sparse + 10M Ngram | 86.57M / 54.45M | 2.7294 | 2.2467 | 4.4493 | 3.1160 | 38.3% / 59.4% | 37.5k | 8.47 GiB |
| Sparse + 25M Ngram | 101.25M / 54.45M | 2.7187 | 2.2349 | 4.4426 | 3.1022 | 22.9% / 69.7% | 33.4k | 8.69 GiB |

For a quick generation check, I sampled 175 continuations from each model. The parse rate checks whether the prompt and generated code can be parsed as Python. The loop rate uses a simple repetition heuristic. Neither measure checks whether the code works. The 25M-memory model had the lowest validation losses; the 10M model had the best results on these two rough generation checks. I need the functional coding tests before drawing a conclusion.

## Results at 250M tokens

| Model | Tokens | Mixed NLL | Code NLL | General NLL | Technical NLL | Functional coding | Parseable / loop | Train tok/s | Peak VRAM | Promotion |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---|
| Dense | 250,003,456 | 2.4840 | 2.0029 | 4.2245 | 2.8501 | Pending matched benchmark | Pending | 40.0k | 7.28 GiB | CONTINUE provisionally; only 250M candidate evaluated so far |

Dense is the first model to finish the 250M evaluation. The other three are still running, so this row is not yet a comparison and is not enough to drop a model. These results use one seed and are research measurements, not a public benchmark.

## Checkpoints and methods

The [scaling table](../results/prompt3/milestone_history.csv) keeps the results from each token count. Each experiment summary records its configuration, data and tokenizer hashes, and checkpoint hash. The weight and resume checkpoints stay on the training machine rather than in Git. The frozen corpus manifest SHA-256 is `501361addc446c234c765d3634e3401446f0b4a7c65217005b5dcd6aa0b7123e`; the tokenizer SHA-256 is `832daf453fc64e28f50df00ac2e458718576bb26e3f16e737f18b120ae4e309a`.

Functional coding tests and the remaining 250M evaluations are still pending. After comparing the models at the same token count, I will stop clear losers and take the remaining candidates to 500M. One model will be selected at that point.

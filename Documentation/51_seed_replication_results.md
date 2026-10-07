# Selected-pair seed replication

Sparse and Sparse + Ngram were compared at four cumulative checkpoints with training seeds 42 and 1337. Data and evaluation order stayed fixed at seed 42; only model-training randomness changed. Lower NLL is better. Checkpoint and input identities are verified in `results/replication_seed1337/verification.json`.

## Matched losses

| Tokens | Seed | Model | Mixed NLL | Code NLL | General NLL | Technical NLL |
|---:|---:|---|---:|---:|---:|---:|
| 20,004,864 | 42 | Sparse | 4.464748 | 3.916070 | 5.969997 | 4.714492 |
| 20,004,864 | 42 | Sparse + Ngram | 4.171885 | 3.623517 | 5.846257 | 4.489081 |
| 20,004,864 | 1337 | Sparse | 4.212787 | 3.665866 | 5.883903 | 4.540365 |
| 20,004,864 | 1337 | Sparse + Ngram | 4.332048 | 3.781472 | 5.925440 | 4.614742 |
| 50,003,968 | 42 | Sparse | 3.276931 | 2.756853 | 5.109978 | 3.672832 |
| 50,003,968 | 42 | Sparse + Ngram | 3.263211 | 2.743938 | 5.074183 | 3.638185 |
| 50,003,968 | 1337 | Sparse | 3.263232 | 2.742787 | 5.110932 | 3.657097 |
| 50,003,968 | 1337 | Sparse + Ngram | 3.252628 | 2.738803 | 5.069121 | 3.637289 |
| 70,000,640 | 42 | Sparse | 3.083776 | 2.579422 | 4.902308 | 3.499052 |
| 70,000,640 | 42 | Sparse + Ngram | 3.078860 | 2.574937 | 4.879346 | 3.472890 |
| 70,000,640 | 1337 | Sparse | 3.082941 | 2.576044 | 4.914545 | 3.487444 |
| 70,000,640 | 1337 | Sparse + Ngram | 3.061471 | 2.555758 | 4.871208 | 3.459445 |
| 100,007,936 | 42 | Sparse | 2.981258 | 2.481819 | 4.775507 | 3.397397 |
| 100,007,936 | 42 | Sparse + Ngram | 2.974749 | 2.476290 | 4.750409 | 3.373976 |
| 100,007,936 | 1337 | Sparse | 2.984690 | 2.482850 | 4.786799 | 3.388244 |
| 100,007,936 | 1337 | Sparse + Ngram | 2.961288 | 2.459072 | 4.740826 | 3.356157 |

## Interpretation

At 20M, the model ordering reverses: seed 42 favors Ngram by 0.292863 mixed nats, while seed 1337 favors Sparse by 0.119261. From 50M through 100M, both seeds favor Ngram. At 100M, the measured mixed/code gain is 0.006509/0.005529 nats for seed 42 and 0.023402/0.023778 for seed 1337. Two seeds on one frozen data order provide a sensitivity check, not a reliable estimate of training-seed variability.

Residual-off ablation worsens Ngram's own loss at each seed-42 checkpoint. The 100M penalties are 0.092697 mixed, 0.154881 code, 0.089014 general, and 0.155034 technical nats. This shows that the trained model uses its memory; it does not by itself show that Ngram will outperform a separately trained Sparse model.

All 12 experts receive held-out traffic at routed layers 2, 5, and 8. Category/expert mutual information remains small (about 0.026–0.040 bits at 100M), and all primary arithmetic-code probes score 0/6. Neither result establishes coding-agent capability or useful semantic specialization.

The verification file binds checkpoints, source/config/data/tokenizer/evaluation identities, and evaluation documents. Compact CSV tables are in the same directory. Large checkpoint payloads remain outside Git and are identified by hashes and sizes.

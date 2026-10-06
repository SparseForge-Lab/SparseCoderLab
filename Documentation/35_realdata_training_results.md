# Research-v1 training results

## Current continuation, 2026-10-06

The shutdown pause ended with the user's Prompt-2.5 request. Dense and Sparse20M/50M are complete. Training is stopped again at the user's explicit choice after GitHub preparation; Ngram is unstarted. The matched all-model50M review and70M/100M remain pending. No final architecture winner exists yet.

| Model | Actual tokens | Mixed NLL | Code NLL | Cumulative step tok/s |
|---|---:|---:|---:|---:|
| DenseCompute | 20,004,864 | 4.302957 | 3.757303 | 30,090 |
| SparseV3 | 20,004,864 | 4.464748 | 3.916070 | 22,528 |
| DenseCompute | 50,003,968 | 3.288592 | 2.778863 | 31,892 |
| SparseV3 | 50,003,968 | 3.276931 | 2.756853 | 23,367 |

Sparse-Dense at20M is+0.161791 mixed/+0.158768 code (lower NLL is better). This early result does not determine the full curve. All immutable source/checkpoint identities remain in the canonical summaries and release snapshots. Cumulative quality-run speeds have different scopes from the controlled Prompt-1 benchmark. CPU playground activity can affect host/wall throughput.

At50M Sparse-Dense is−0.011661 mixed/−0.022011 code, so the signed gap changed from a Sparse deficit to a small advantage. General difference is−0.003311 nats/token. The cumulative step-time cost ratio is1.365x Dense, close to the separate controlled Prompt-1 result1.376x. This single-seed sign change motivates completing the remaining curves and considering replication; it is not a final promotion or large-model claim. Sparse50M checkpoint SHA2c9ec686cc7213a80b469c5f21fd5b90545d1ff02c3e4bfd20bbd97baac49954.

Dense20M/50M micro-code checks passed0/6 each. Better prediction loss has not demonstrated useful coding or chat capability. Manual playground prompts agree with those limitations but are not a new controlled evaluation.

## Historical shutdown state

User-requested pause: Dense20M and50M are complete. At50,003,968 tokens the mixed NLL is3.288592 and code diagnostic NLL2.778863; step throughput31,892 tok/s. The Dense20M/50M cumulative wall measurements are lower bounds following the reporting-only recovery. The50M milestone/rolling checkpoint pair is verified on CPU and training has stopped for shutdown. No matched sparse/Ngram result exists yet, so no architecture conclusion is available. Resume the full original comparison after the user returns.

Corpus freeze and real-stream resume/readiness gates passed. Primary cumulative quality training is running; milestone quality remains pending. No architecture winner or quality gap is asserted yet. The cumulative endpoints/protocol are predeclared in Documentation34; results/research_v1/training_comparison.csv, training_curves.csv and milestone_deltas.csv will contain executed observations only.

Every milestone must include mixed/code/general/technical and sufficiently populated language losses, bits/token, actual consumed tokens, step/wall rates, allocated memory, exact stored and estimated active parameters/FLOPs, immutable checkpoint identities and provenance. Evaluate sparse-minus-dense, Ngram-minus-dense and Ngram-minus-sparse at all four endpoints; classify growing/shrinking/oscillating/sign-changing/stable gaps. Paired document intervals describe validation sampling only and do not replace seed replication or account fully for repository correlation.

Original six-task arithmetic Python checks are bounded greedy48-token evaluations using a restricted numeric AST interpreter. Generated code is never executed via Python exec or shell. They can reveal syntax/basic functional progress but do not establish useful coding-agent performance or benchmark leadership. All task prompts/hashes, generated text and outcomes are retained per milestone.

At100M decide whether seed1337 replication of the strongest relevant pair is warranted by ambiguity, trend and sampling evidence. No automatic second-seed all-model schedule or DenseSize control should delay primary completion. All final questions, promotion/hold/redesign logic and scaling limits must be answered from measured curves, not this preparatory note.

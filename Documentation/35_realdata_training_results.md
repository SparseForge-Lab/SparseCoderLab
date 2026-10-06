# Research-v1 training results

## Current continuation, 2026-10-06

Dense/Sparse20M/50M and Ngram20M are complete. Ngram20M→50M is running cumulatively with Roblox kept open. All-three50M review,70M/100M, replication decision and final classification remain pending.

| Model | Actual tokens | Mixed NLL | Code NLL | General NLL | Technical NLL | Cumulative step tok/s |
|---|---:|---:|---:|---:|---:|---:|
| dense | 20,004,864 | 4.302957 | 3.757303 | 5.917018 | 4.588342 | 30,090 |
| sparse | 20,004,864 | 4.464748 | 3.916070 | 5.969997 | 4.714492 | 22,528 |
| memory | 20,004,864 | 4.171885 | 3.623517 | 5.846257 | 4.489081 | 22,217 |
| dense | 50,003,968 | 3.288592 | 2.778863 | 5.113289 | 3.684617 | 31,892 |
| sparse | 50,003,968 | 3.276931 | 2.756853 | 5.109978 | 3.672832 | 23,367 |

Sparse-Dense mixed/code gap changes from+0.161791/+0.158768 at20M to−0.011661/−0.022011 at50M. Ngram20M improves over Sparse by−0.292863/−0.292553 and over Dense by−0.131072/−0.133786. Its zero-residual ablation penalty is+0.059097 mixed/+0.094661 code/+0.035774 general/+0.109459 technical. See Documentation44 for sampling/routing/memory evidence and limits. All three20M probes and the existing Dense/Sparse50M probes passed0/6; useful coding capability is unproved.

Quality-run speed is cumulative observational timing, affected by CPU playground and later Roblox concurrency; it is not the controlled Prompt-1 study. Dense cumulative wall remains a reporting-recovery lower bound. No final promotion or scaling claim.

## Historical shutdown state

User-requested pause: Dense20M and50M are complete. At50,003,968 tokens the mixed NLL is3.288592 and code diagnostic NLL2.778863; step throughput31,892 tok/s. The Dense20M/50M cumulative wall measurements are lower bounds following the reporting-only recovery. The50M milestone/rolling checkpoint pair is verified on CPU and training has stopped for shutdown. No matched sparse/Ngram result exists yet, so no architecture conclusion is available. Resume the full original comparison after the user returns.

Corpus freeze and real-stream resume/readiness gates passed. Primary cumulative quality training is running; milestone quality remains pending. No architecture winner or quality gap is asserted yet. The cumulative endpoints/protocol are predeclared in Documentation34; results/research_v1/training_comparison.csv, training_curves.csv and milestone_deltas.csv will contain executed observations only.

Every milestone must include mixed/code/general/technical and sufficiently populated language losses, bits/token, actual consumed tokens, step/wall rates, allocated memory, exact stored and estimated active parameters/FLOPs, immutable checkpoint identities and provenance. Evaluate sparse-minus-dense, Ngram-minus-dense and Ngram-minus-sparse at all four endpoints; classify growing/shrinking/oscillating/sign-changing/stable gaps. Paired document intervals describe validation sampling only and do not replace seed replication or account fully for repository correlation.

Original six-task arithmetic Python checks are bounded greedy48-token evaluations using a restricted numeric AST interpreter. Generated code is never executed via Python exec or shell. They can reveal syntax/basic functional progress but do not establish useful coding-agent performance or benchmark leadership. All task prompts/hashes, generated text and outcomes are retained per milestone.

At100M decide whether seed1337 replication of the strongest relevant pair is warranted by ambiguity, trend and sampling evidence. No automatic second-seed all-model schedule or DenseSize control should delay primary completion. All final questions, promotion/hold/redesign logic and scaling limits must be answered from measured curves, not this preparatory note.

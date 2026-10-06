# Phase 1B results — not run

Stage C1 10M-token elimination, C2 30M-token survivors and seed 1337 replications did not start. No real corpus/tokenizer was frozen. The research lanes contain readiness notes, not runnable training configurations. The result file results/phase1b_comparison.csv explicitly labels all three intended primary models `not_run_stop_review`; blank metrics mean unmeasured, not zero loss or zero performance.

The user's stop instruction is: "sparse model is dramatically slower without measurable quality benefit". At the completed matched Phase 1A endpoint the observed wall penalties are 2.67x for SparseV3 and 2.85x with memory, while mixed synthetic NLL differences are negligible and oscillate across checkpoints. This warrants a review stop. It does not establish how these architectures rank on real code.

Quality per active compute/stored parameter/wall time on real data, code/general real validation, routing specialization/locality, seed reproducibility and the 30-minute C1 feasibility are unknown. No survivors were promoted. DenseSize was not run. No MTP, RouteAhead, memory-capacity sweep, long context, compaction, physical offload, TPU/JAX or repository RL training occurred.

Recommended next experiment is **architecture redesign/review**, focused first on the actual eager expert-dispatch cost and whether the memory residual contributes useful aggregate information. Design any revised implementation and matched controls before resuming a verified real-corpus comparison. This is a proposal for review; it has not started. Longer core pretraining, Top1-vs-Top2, MTP-3 and compaction lack evidence for automatic promotion here.

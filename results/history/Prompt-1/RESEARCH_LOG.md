# Research log — 2026-10-05

1. Probed hardware before heavy installation. WSL2 cannot start due to disabled virtualization; selected native Windows. Created isolated project-local Python 3.12 venv and official torch 2.14.1+cu130. Driver/toolkit/global Python and existing projects untouched.
2. Executed BF16 causal SDPA backward on RTX 5070; verified SM120 wheel coverage. D: is SATA, not NVMe. Added unbuffered Windows benchmarks because initial cached reads would misrepresent storage bandwidth.
3. Implemented shared RMSNorm/RoPE/GQA/SDPA/SwiGLU backbone, dense size/compute configs, free-routing Top1/Top2, optional causal N-gram memory and three-step shared MTP.
4. Exact counts show SparseV3 is 21.56256M without memory and 25.767425M with memory. Did not alter architecture to force an expected number.
5. Generated original CC0 bounded dev documents, content-hash split/dedup, exact 32,768 BPE trained on train split only. No dataset/model download. Held-out real-code quality remains unmeasured.
6. Core/lab tests executed; DenseCompute overfit reduced loss 10.415672 -> 0.000174849 in 80 steps. Five-minute baseline profiler and sparse smoke tests are engineering readiness checks.
7. Added archived exact blocks, repeated-compaction fixtures, graph packing, tier-cache simulation, route prediction interface, large capacity sweeps and disposable toy trajectories. These do not establish architecture benefits.
8. User requested a Documentation folder with separate files and a tracked goal; created both while retaining the stop-before-long-training constraint.

Machine-readable results and FINAL_STATUS remain the authoritative evidence. No architecture winner is selected from smoke runs.

9. Trace integration exposed a verification bug under `torch.no_grad`; wrapped the BF16 backward probe in `torch.enable_grad`, added a GPU regression and reran the trace collector successfully.
10. Supplied Strata launcher was inspected read-only. Native one-shot capture exited 2 because its IQ pack requires speculative windows/prefill; token-position alignment is absent in those dumps. Skipped the optional real trace dataset, preserved diagnostics and changed no existing installation.
11. Shared-MTP greedy verification matched exact base output but the tiny smoke decode was slower (~0.80x in one eight-token observation). No useful MTP speedup is established. RouteAhead and compaction measurements retain their synthetic/oracle caveats.

12. Continuation began with the unchanged-source Phase 0 gate: 30 tests passed. Preserved previous status, measurements, gate/tests and leaderboard under results/history/pre_phase1. Added sampled memory diagnostics and an evaluation-only zero-residual ablation with meaningful regression coverage; 32 tests passed before matched training.
13. Predeclared fresh seed-42 endpoint as 611 full updates / 5,005,312 tokens. All three primary architectures completed that exact endpoint with identical tokenizer, synthetic shards/order, training policy and evaluation steps. Archived the byte-exact source and retained raw metrics/configs/provenance/checkpoints.
14. SparseV3/Ngram wall costs were 2.67x/2.85x dense, while mixed NLL differences were negligible. Same-checkpoint Ngram ablation changed mixed NLL by +0.000054, with a weak code-subset effect. Applied the user's explicit slowdown review condition: stopped before research-data acquisition, Phase 1B, new representative traces, capacity training and other later phases. Separate documentation 23–27 and explicitly not-run Phase 1B CSV preserve the distinction between measured, calculated and unknown evidence.

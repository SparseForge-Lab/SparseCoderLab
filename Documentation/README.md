# Documentation index

New permanent continuation documents:38_release_and_model_card_evidence.md covers research ledger/release/card/timeline rules;39_local_dense_playground.md covers the explicitly requested loopback Dense50M CPU playground. Exact Prompt-2.5 request:00_prompt_2_5_request.md. All remain part of Prompt-2.

This directory is the maintained documentation for SparseCoderLab. The small models test architecture hypotheses; they are not expected to be useful coding agents.

The `00_*` exact request records mentioned below are local-only provenance notes and are intentionally excluded from the public GitHub repository. Public protocols and measured reports remain available.

| File | Contents |
|---|---|
| 01_environment.md | Hardware, GPU verification, Python isolation and storage measurements |
| 02_installation.md | Reproduce the Windows/WSL environment and dependency lock |
| 03_tokenizer.md | Frozen byte BPE, reserved markers and quality checks |
| 04_data.md | Sources, licensing, deterministic splits, bounded preprocessing |
| 05_models.md | Shared backbone, exact parameter arithmetic and fair comparisons |
| 06_ngram_memory.md | Causal deterministic hashing, gradients and diagnostics |
| 07_moe_routing.md | Free routing, balance loss and routing traces |
| 08_expert_atlas.md | Graph construction and post-training physical packing |
| 09_cache_simulator.md | Three-tier model, bandwidth, cache policies and assumptions |
| 10_mtp.md | Shared three-step draft and greedy speculative verification |
| 11_routeahead.md | Future-route predictors and prefetch success criteria |
| 12_context_compaction.md | Exact archive, repeated compaction and model evaluation |
| 13_training_resume.md | Timer, checkpoint contents, pause/resume and provenance |
| 14_experiment_schedule.md | Successive halving and mandatory phase gates |
| 15_evaluation.md | Metrics, fairness, uncertainty and missing measurements |
| 16_large_projection.md | Large candidate arithmetic and memory capacity sweep |
| 17_tpu_and_attention.md | Later TPU implementation and isolated mixer research |
| 18_agentic_lane.md | Disposable toy repositories, verified trajectories and loops |
| 19_testing.md | Actual test commands and regression coverage |
| 20_measurements.md | Generated measurements and calculated training duration |
| 21_limitations.md | Implemented reference code versus unresolved research |
| 22_strata_adapter.md | Read-only inspection and optional trace interchange |

`../FINAL_STATUS.md` records the current runtime decision after the historical Phase1A review stop. `../RESEARCH_LOG.md` records implementation decisions. Machine-readable evidence lives in `../results/`; checkpoints live in `../experiments/`. File20 remains historical Phase0 evidence; its original copy is also preserved under results/history/pre_phase1.

| File | Topic |
|---|---|
| 00_phase1_request.txt | Exact continuation request and stop conditions |
| 23_phase1a_protocol.md | Predeclared matched endpoint and measurement scope |
| 23_phase1a_results.md | Executed training comparison, fairness, timing and review stop |
| 24_research_corpus.md | Candidate source review and explicitly unprepared real corpus |
| 25_phase1b_results.md | Explicit not-run states and next experiment recommendation |
| 26_phase1a_ngram.md | Gate, gradients, sampled buckets and trained-checkpoint ablation |
| 27_phase1_routing_review.md | Observed balance versus unmeasured specialization/locality |
| 00_moe_runtime_request.txt | Exact Prompt-1 request and stop conditions |
| 28_moe_runtime_optimization.md | Reference/grouped profiling, actual speed, geometry and Case A decision |
| 29_prompt_archives.md | Mandatory future goal/numbered SHA256 second-copy archive convention |
| 30_moe_backend_correctness.md | Optimizer/checkpoint semantics, parity tolerances, rejected trials and controls |
| 31_moe_benchmark_protocol.md | Warmed layer/full-model/geometry/Ngram scopes, provenance and reproduction |

| File | Topic |
|---|---|
| 00_research_v1_request.txt | Exact Prompt-2 real-data continuation request |
| 32_research_v1_corpus.md | Verified bounded sources, split/dedup audit and frozen corpus |
| 33_research_tokenizer.md | Measured real-data tokenizer decision and freeze |
| 34_20m_50m_70m_100m_protocol.md | Matched cumulative updates, resume gates and evaluation scope |
| 35_realdata_training_results.md | Primary quality/cost curves and replication decision |
| 36_sparse_specialization.md | Held-out category/language routes and enrichment |
| 37_ngram_realdata_results.md | Gate/table diagnostics and same-checkpoint ablations |
| 00_prompt_2_5_request.md | Exact verified continuation request from the prompt pack |
| 38_release_and_model_card_evidence.md | Permanent ledger, release, model-card and figure rules |
| 39_local_dense_playground.md | CPU-only browser playground and observed capability limits |
| 41_primary_run_controls.md | Sequential phase barriers, process guard, thermal samples and final audit |
| 42_github_repository.md | GitHub, Apache-2.0, public/private artifact policy and publication checks |

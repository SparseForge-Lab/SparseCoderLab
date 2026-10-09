# Documentation index

This directory contains the project's research protocols, implementation notes, and measured results. The models are small research systems; their language-modeling scores do not establish coding-agent capability.

Exact internal task prompts and local workflow records are kept out of the public repository. Public reports describe the experiments and their measured evidence.

| File | Contents |
|---|---|
| 01_environment.md | Hardware, GPU verification, Python isolation, and storage measurements |
| 02_installation.md | Reproduce the Windows/WSL environment and dependency lock |
| 03_tokenizer.md | Frozen byte BPE, reserved markers, and quality checks |
| 04_data.md | Sources, licensing, deterministic splits, and bounded preprocessing |
| 05_models.md | Shared backbone, parameter arithmetic, and fair comparisons |
| 06_ngram_memory.md | Causal deterministic hashing, gradients, and diagnostics |
| 07_moe_routing.md | Free routing, balance loss, and routing traces |
| 08_expert_atlas.md | Graph construction and post-training physical packing |
| 09_cache_simulator.md | Three-tier model, bandwidth, cache policies, and assumptions |
| 10_mtp.md | Shared three-step draft and greedy speculative verification |
| 11_routeahead.md | Future-route predictors and prefetch success criteria |
| 12_context_compaction.md | Archive, repeated compaction, and model evaluation |
| 13_training_resume.md | Timer, checkpoint contents, pause/resume, and provenance |
| 14_experiment_schedule.md | Successive halving and mandatory phase gates |
| 15_evaluation.md | Metrics, fairness, uncertainty, and missing measurements |
| 16_large_projection.md | Large candidate arithmetic and memory capacity sweep |
| 17_tpu_and_attention.md | Later TPU implementation and isolated mixer research |
| 18_agentic_lane.md | Disposable toy repositories, verified trajectories, and loops |
| 19_testing.md | Actual test commands and regression coverage |
| 20_measurements.md | Generated measurements and calculated training duration |
| 21_limitations.md | Implemented reference code and unresolved research |
| 22_strata_adapter.md | Read-only inspection and optional trace interchange |
| 23–31 | Earlier architecture, runtime, and archive research |
| 32_research_v1_corpus.md | Verified bounded sources, split/dedup audit, and frozen corpus |
| 33_research_tokenizer.md | Measured real-data tokenizer decision and freeze |
| 34_20m_50m_70m_100m_protocol.md | Matched cumulative updates, resume gates, and evaluation scope |
| 35_realdata_training_results.md | Primary quality/cost curves and replication decision |
| 36_sparse_specialization.md | Held-out category/language routes and enrichment |
| 37_ngram_realdata_results.md | Gate/table diagnostics and same-checkpoint ablations |
| 38_release_and_model_card_evidence.md | Research ledger, releases, model-card evidence, and figure rules |
| 39_local_dense_playground.md | CPU-only browser playground and observed capability limits |
| 44_matched_20m_evidence.md | First all-three real-data comparison and its limits |
| 46_public_repository_cleanup.md | Public/private classifications and experiment ignore rules |
| 50_microbatch_throughput.md | Fixed-effective-batch throughput measurements and limits |
| 51_seed_replication_results.md | Matched second-seed results and sensitivity evidence |
| 54_realdata_comparison.md | Final loss comparison, interpretation, and evidence links |
| 54_prompt3_milestone_results.md | Four-model 100M comparison and later exploratory checkpoints |
| 55_repository_data_pipeline.md | Pinned repository ingestion, source review, and measured preprocessing fixture |
| 56_multilingual_repository_pilot.md | Thirteen-repository token/language inventory, FIM and remaining data gates |
| 57_safe_functional_evaluation.md | Completed saved-completion scoring, WASI limits, provenance and overlap screen |
| 58_implementation_correctness_and_memory.md | Optimizer/resume fixes, measured VRAM, parity, streamed data and remaining scaling limits |
| 59_repository_transition_preparation.md | Pinned source preparation and explicit phase-transition policy |
| 60_frozen_repository_transition_preflight.md | Frozen corpus, measured recipe/coverage, canonical functional evidence and exact CUDA transition resume |
| 61_matched_repository_training.md | Completed common 250M repository-data campaign, recovery checks and matched results |
| 62_unified_cpu_functional_evaluation.md | Deterministic 100-prompt functional scoring, routing/N-gram telemetry, safe checkpoint queue, and architecture estimates |

Machine-readable results live under `results/`. Large datasets and checkpoints are not included in ordinary Git; their manifests retain integrity metadata.

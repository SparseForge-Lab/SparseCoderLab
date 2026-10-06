# Evidence and interpretation

Prompt-2.5 continuation evidence is maintained in `research_history.csv` (append-only verified checkpoint rows), `model_card_evidence.json` (measured/calculated/planned/unknown), `research_timeline.json`, local `releases/{20M,50M,70M,100M}` snapshots and `figures/` generated from canonical CSV/JSON. Partial/planned snapshots are not completed comparisons or published HF/GitHub releases. Exact resume verification lives in `research_v1/resume_state.json` and `resume_verification.json`. Current46-test source-bound evidence supersedes historical40-test engineering gates; historical files are preserved. The local Dense playground is optional CPU inference and does not establish useful coding/chat capability or alter frozen evaluation.

Prompt-2 completed quality rows are appended to `leaderboard.csv` without rewriting historical bytes. Their `tok_s` is cumulative wall throughput (Dense has a documented lower-bound wall time), while `training_comparison.csv` separately retains cumulative training-step throughput. Prompt-2 router entropy is a token-weighted held-out category/layer mean, unlike the historical Phase1A sampled training-microbatch aggregation. Context, tokenizer, corpus and measurement scopes differ between historical phases; do not compare their absolute NLL as a common benchmark.

- environment_before.json: initial probe; cached I/O is labelled, not real-device bandwidth.
- environment.json: isolated environment and unbuffered 4 MiB SATA SSD measurements.
- gpu_verification.json: actual SM120 / CUDA / BF16 SDPA backward.
- tests.xml + test_output.txt: latest complete executed regression suite.
- overfit.json: actual DenseCompute learning on the tiny repeating fixture.
- profile_DenseCompute.json: five-minute standalone microbatch2 profiler; SparseV3/Ngram profiles are 30 seconds.
- profile_resume_before/after.json: bounded profiler continuation; not replacements for the five-minute profile.
- parameter_counts.json: unique actual trainable weights and declared active estimates.
- tokenizer_quality.json + data/manifest.json: exact BPE roundtrips, language tokenization and licensed data/split provenance.
- routes.jsonl, cache_simulation.json: micro-model actual routes, held-out atlas/cache experiments. SSD residual latency was set to zero when using measured 4 MiB throughput to avoid counting transfer time twice.
- scaled_simulation.json, large_projection.json: assumption-driven synthetic topology/projection; not large-model performance measurements.
- mtp_evaluation.json: eight-token reference greedy speculation, exact parity; tiny checkpoint and no KV cache.
- experiments/routeahead_smoke/summary.json: ten-step train/val predictor with simulated prefetch; no stall benefit observed.
- context_fixtures.json and context_model_eval.json: oracle archive retention and separate neural answer-NLL smoke. Neither is generated task pass rate.
- agent_trajectory.json: scripted failed/repeated/repaired verified trajectory, not autonomous-model success.
- strata_status.json + strata_capture.log: optional native capture failure and read-only skip rationale.
- phase0_gate.json: current-source engineering gate; does not launch a later phase.

Leaderboard rows are observed smoke or cumulative resumed runs. They are not independent research replications. Early rows' tok_s used train-step time; later summaries explicitly separate train_step_tok_s and wall tok_s, and wall time includes checkpoint/evaluation. Prefer the equally sized timing_* summaries for matched smoke observations and timer_before_resume.json for the calculated first-run duration. Early MTP FLOPs used unique-active 6N; later runs include repeated draft-block and extra output-head cost. No row is a measured hardware FLOP count. Source hashes/git-dirty state preserve provenance for work-in-progress runs.

All timing predictions use recorded throughput; no absolute coding benchmark or final architecture winner was inferred.

Phase 1A adds three fresh 5,005,312-token runs under experiments/phase1a_*. phase1a_comparison.csv and phase1a_curves.csv retain measured losses/timings and calculated parameter/FLOP fields. phase1a_diagnostics.json aggregates sampled router/memory observations; phase1a_ngram_ablation.json zeroes the residual on the same trained checkpoint without retraining. phase1a_integrity.json verifies matched configs, endpoint/cursor/hashes and frozen shard checksums. phase1_review_decision.json records the explicit stop before real-data acquisition or Phase 1B. phase1b_comparison.csv contains not-run statuses and blank measurements. research_source_review.json is provider API metadata only, not acquired dataset content.

history/pre_phase1 preserves the previous status/test/leaderboard evidence. history/phase1a_source.zip and phase1a_source_manifest.json preserve the byte-exact source used by the runs; postprocessing tools were added afterward. Phase 1A leaderboard router_entropy aggregates 61 logged last-microbatch observations per model. The synthetic loss gaps do not establish a quality advantage, and observed wall costs of 2.67x/2.85x triggered the user's review stop.

Prompt-1 runtime evidence is separate: moe_reference_profile.json was captured before optimization; moe_grouped_profile.json profiles the final backend. moe_backend_parity.json is source-bound deterministic correctness evidence; moe_reference_repeatability*.json record baseline controls. history/Prompt-1 retains rejected padded-backward source/parity/timing, initial import failure and pre-prompt status/gate/test/leaderboard copies. Failed reports/output are explicitly historical trial evidence, not current gate failures. moe_regression_tests.* and moe_phase0_output.txt record the final40-test passing suite and current-source gate. tests.xml/test_output.txt now also hold that40-test gate; their previous copies are retained.

moe_microbench.csv, moe_fullmodel_benchmark.csv and moe_expert_granularity.csv contain actual RTX5070 measurements; per-model/repetition benchmark JSONL files preserve raw short wall-loop observations. moe_runtime_decision.json aggregates them, moe_runtime_report_output.txt mirrors aggregation, and moe_benchmark_provenance.json binds source/config/data/CSV hashes. Runtime wall scopes exclude initialization/evaluation/checkpoint costs; they do not replace historical Phase1A wall rates. No validation quality values or speed-only rows were added to leaderboard.csv. See Documentation28–31 and GOALS/Prompt-1.md. prompt-1_archive_manifest.json enumerates final canonical changed bytes; prompt-1_archive_verification.json records final ZIP/manifest/canonical-file checks. Archive: ~\Data-Zip\Prompt-1\Prompt-1.zip.


Prompt-2 research_v1: pinned source metadata/download verification, rejected preprocessing reports, final corpus/tokenizer manifests/statistics,46-test evidence, exact-stream resume verification and readiness gate. experiments/research_v1/{dense,sparse,memory} holds fresh cumulative seed42 metrics and immutable milestones. Frozen evaluation_index defines mixed/category/language scopes. Full quality, specialization, ablation and micro-code results are emitted per matched endpoint; no primary quality conclusion exists until those files are complete. Historical Prompt-1 remains separately archived.

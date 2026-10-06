# Prompt-1 — MoE execution overhead

Status: complete after final archive verification receipt. Archive: ~\Data-Zip\Prompt-1\Prompt-1.zip. Canonical project: D:\SparseCoderLab. Number chosen by inspecting Data-Zip at the beginning: no existing numbered archives, therefore Prompt-1.

## Exact objective

FIX AND CHARACTERIZE MoE EXECUTION OVERHEAD. Preserve SparseV3's mathematical behavior while replacing or supplementing eager experts with efficient GPU execution. Profile first; retain the reference numerical oracle; gate grouped execution on FP32/BF16 forward/routing/gradient/optimizer/checkpoint parity; measure warmed layer/full-model/granularity/current Ngram runtime on RTX5070; document the decision and freeze every prompt change/evidence in a verified SHA256 second-copy archive.

## Starting state — historical context only

Starting commit 0e36357925a6b4d0f3fa847adb749766214b3c34. Phase1A had three completed fresh 5,005,312-token synthetic runs. Historical step rates: DenseCompute30324, SparseV310874, Ngram10174 tok/s, sparse cost2.79x dense. Tiny loss gaps established no quality winner. Expert geometry was12x160 over three width320 capacity layers, Top1 selected-probability multiplier, no drops. Original checkpoints, tokenizer, shard order and configs remain intact. Exact request is Documentation/00_moe_runtime_request.txt; beginning-of-prompt byte hashes and old status/gate/leaderboard/tests are retained under results/history/Prompt-1.

## Explicit non-goals

No model/Ngram research redesign, real-corpus acquisition,20M/50M/70M/100M quality training, second quality seed, Top2 quality, MTP/RouteAhead, compaction, custom attention, long-context scaling, physical offload, TPU/JAX or repository RL. No global dependency/CUDA changes, mandatory exotic kernel, subagents or automatic next phase. Strata launcher/installation untouched. Brain MCP unavailable.

## Success criteria and stop conditions

Profile before optimization; retain reference; lossless grouped Top1 math and checkpoint compatibility; justified numerical parity across both precisions and several optimizer updates; all regressions pass; measured warmed layer/full-model and controlled12/6/4 geometry plus Ngram cost; honest CaseA/B/C; finalized docs, goal, exact changed-file manifest and verified ZIP while retaining canonical files. Failed correctness blocks promotion, no silent CPU fallback or token truncation, bounded measurements and no next long phase. These criteria are satisfied by the final backend; rejected intermediate trials are explicitly retained.

## Experiments and measured results

1. Focused original reference profile: trained checkpoint,1024 context/2 microbatch/4 accumulation,3 warm+3 recorded updates.432 expert forwards/nonzero calls; last-group range68–247, mean170.67. Reference CPU forward1599.31ms/backward2200.46ms/clipopt192.29ms;42729 runtime launches. Expert GEMM scope CPU296.29ms/device5.61ms. Host indexing/launch/sync overhead was a major cause.
2. Optional stable GPU sorted/grouped packing, observed max capacity rounded16, no drops; batched forward/input gradients with unpadded weight reductions. Separate registered expert parameters/state names/order remain unchanged. GroupedAdamW restores None gradients for experts absent across accumulated microbatches before ordinary AdamW, preserving momentum/decay/step counters. Top2 remains reference-only.
3. Five-update FP32 CPU/fresh BF16 RTX/trained-checkpoint BF16 deterministic parity, every routing/output/gradient/post-update tensor reported. All final tested differences zero, unchanged declared tolerances. Old AdamW state loads, state_dict reference->grouped->reference roundtrips.40 regressions pass; current-source Phase0 gate passes (historical overfit/five-minute profile are reused readiness evidence, not rerun quality).
4. Final96-row warmed layer matrix: contexts256/1024/2048,batches1/2/4/8,four distributions,two backends. Natural f+b speedup median3.458x,range3.099–3.761;ctx1024/batch2f+b26.199->7.043ms (3.720x),forward13.250->3.117ms(4.252x). Strong-skew minimum1.766x. Batch8 largest configured tested candidate, not absolute safe maximum; no OOM. Fixture-inclusive memory and unreliable short GPU-utilization are disclosed.
5. Two reverse-order matched full-model repetitions, twenty measured updates per separate pure and wall loop. Aggregate step/wall rates: dense25272.859/25567.751;reference11265.972/11274.647;grouped18360.653/18289.225;Ngram17792.161/17650.617 tok/s. Grouped speedup1.630x step/1.622x wall versus reference;cost1.376x/1.398x dense. Wall excludes initialization/evaluation/checkpoint I/O and is not substituted into old Phase1A measurements.
6. Current Ngram incremental cost+3.195% step time/+3.618% wall time,+69141504 allocated bytes(65.94MiB). No redesign/capacity sweep or new quality claim.
7. Fresh seed42 geometry smoke holds routed stored capacity5529600 (E*FFN1920);12x160/6x320/4x480 step rates18618.066/20370.365/20326.575. Stored counts21562560/21556800/21554880,active16493760/16948800/17407680. Six/four effectively tied:0.22% pure-step gap;four wins wall. Finite losses fall ~10.4->9.94–9.96. No automatic geometry change or quality ranking.
8. Final grouped profile: CPU forward875.51ms/backward1283.61ms/clipopt183.27ms;runtime launches25818,CUDA events29964 versus49923 reference. Kernel self250.76ms versus277.61ms reference. Reduced host overhead is the main gain; eager backward, copies/casts/stacking, shared backbone/SDPA/vocabulary softmax and unpadded weight-gradient loops remain. Additional compile/fusion/Triton lanes were unnecessary and not tested.

## Failures, rejected trials and diagnostic repairs

- Initial profiler aggregation merged CPU/CUDA annotation keys and double-counted device totals. Fixed annotation filtering and regenerated concise reference evidence before optimization; no model math changed.
- New regression imports exposed the pre-existing model/router circular import. Initial failure output retained; lazy public model export fixed it, all40 tests pass.
- Initial native trained-checkpoint parity failed, as did native reference-vs-reference repeatability (fifth-step max logit1.6875). Deterministic reference control gave zero differences; the precise nondeterministic kernel was not isolated.
- Fully padded BMM weight backward still failed strict deterministic trained continuation (fifth-step max logit2.6875) despite tiny first-step gradients. Rejected, with source/parity/microbench preserved under history/Prompt-1. Final unpadded weight reductions pass without widening tolerance; final timing matrix was rerun on corrected source.
- Adding metadata to the parity report briefly caused a summary-print KeyError after the successful report had been written. Fixed reporter dispatch and reran successfully; this was output formatting, not a relaxed correctness check.
- Preliminary logs/results outside history are labelled in Documentation30/results README; final source-bound moe_backend_parity.json, deterministic output and final benchmark CSVs are authoritative. No failed version is promoted.

## Final decision and unresolved work

Qualified CaseA: runtime mostly fixed enough for a separately authorized real-data cumulative20M->50M->70M->100M quality experiment.1.38–1.40x dense cost is potentially acceptable, weaker than the strong<=1.35 band. No next phase was started. Keep the12-expert research baseline until controlled quality evidence authorizes a change. Remaining numerical repeatability/skew padding/eager overhead,real quality,tokenizer suitability,multiple seeds,memory benefit,routing locality and scale transfer are unresolved. Historical quality measurements/leaderboard remain unchanged. Documentation28–31, FINAL_STATUS and machine-readable decision/provenance explain scopes and evidence.

## Archive finalization

All canonical files remain under D:\SparseCoderLab. The second-copy ZIP at ~\Data-Zip\Prompt-1\Prompt-1.zip is refreshed after this goal and all documentation are final. External MANIFEST.json and embedded manifest match; GOAL.md mirrors this file. tools/archive_prompt.py opens the final ZIP, verifies CRC and every listed path/SHA256 against current canonical bytes, verifies the goal mirror, and writes results/prompt-1_archive_verification.json plus external VERIFICATION.json. No older archive is overwritten; the current prompt's owned archive may be refreshed. Three large existing Phase1A checkpoints are references with size/SHA256/origin, not copied. Administrative manifest/receipt self-references are excluded from their own inventory to avoid circular hashes. Expected future convention is Documentation29/GOALS README. The receipt is required for completed status.

## Files created/modified by this prompt

The list below covers canonical prompt changes against the starting hash manifest, including historical copies and raw evidence. Final SHA256/path/size inventory is results/prompt-1_archive_manifest.json and the embedded/external MANIFEST.json. Unchanged Phase1A source/config/data/checkpoints and quality leaderboard are excluded from changed files.

- `Documentation/00_moe_runtime_request.txt`
- `Documentation/19_testing.md`
- `Documentation/28_moe_runtime_optimization.md`
- `Documentation/29_prompt_archives.md`
- `Documentation/30_moe_backend_correctness.md`
- `Documentation/31_moe_benchmark_protocol.md`
- `Documentation/README.md`
- `FINAL_STATUS.md`
- `GOALS/Prompt-1.md`
- `GOALS/README.md`
- `RESEARCH_LOG.md`
- `configs/runtime/dense.yaml`
- `configs/runtime/experts4.yaml`
- `configs/runtime/experts6.yaml`
- `configs/runtime/sparse_grouped.yaml`
- `configs/runtime/sparse_grouped_memory.yaml`
- `configs/runtime/sparse_reference.yaml`
- `results/README.md`
- `results/history/Prompt-1/FINAL_STATUS.md`
- `results/history/Prompt-1/RESEARCH_LOG.md`
- `results/history/Prompt-1/grouped_padded_backward.py`
- `results/history/Prompt-1/initial_import_failure.txt`
- `results/history/Prompt-1/leaderboard.csv`
- `results/history/Prompt-1/moe_microbench_padded_backward.csv`
- `results/history/Prompt-1/phase0_gate.json`
- `results/history/Prompt-1/starting_manifest.json`
- `results/history/Prompt-1/test_output_before.txt`
- `results/history/Prompt-1/tests_before.xml`
- `results/history/Prompt-1/trained_parity_deterministic_padded_failure.json`
- `results/history/Prompt-1/trained_parity_initial_failure.json`
- `results/moe_backend_parity.json`
- `results/moe_benchmark_provenance.json`
- `results/moe_dense_rep1_benchmark.jsonl`
- `results/moe_dense_rep2_benchmark.jsonl`
- `results/moe_expert_granularity.csv`
- `results/moe_expert_granularity_output.txt`
- `results/moe_fullmodel_benchmark.csv`
- `results/moe_fullmodel_benchmark_output.txt`
- `results/moe_geometry12_rep1_benchmark.jsonl`
- `results/moe_geometry12_rep2_benchmark.jsonl`
- `results/moe_geometry4_rep1_benchmark.jsonl`
- `results/moe_geometry4_rep2_benchmark.jsonl`
- `results/moe_geometry6_rep1_benchmark.jsonl`
- `results/moe_geometry6_rep2_benchmark.jsonl`
- `results/moe_grouped_profile.json`
- `results/moe_grouped_profile_output.txt`
- `results/moe_grouped_rep1_benchmark.jsonl`
- `results/moe_grouped_rep2_benchmark.jsonl`
- `results/moe_memory_rep1_benchmark.jsonl`
- `results/moe_memory_rep2_benchmark.jsonl`
- `results/moe_microbench.csv`
- `results/moe_microbench_output.txt`
- `results/moe_parity_deterministic_output.txt`
- `results/moe_parity_output.txt`
- `results/moe_phase0_output.txt`
- `results/moe_reference_profile.json`
- `results/moe_reference_profile_output.txt`
- `results/moe_reference_rep1_benchmark.jsonl`
- `results/moe_reference_rep2_benchmark.jsonl`
- `results/moe_reference_repeatability.json`
- `results/moe_reference_repeatability_deterministic.json`
- `results/moe_reference_repeatability_deterministic_output.txt`
- `results/moe_reference_repeatability_output.txt`
- `results/moe_regression_tests.txt`
- `results/moe_regression_tests.xml`
- `results/moe_runtime_decision.json`
- `results/moe_runtime_report_output.txt`
- `results/moe_runtime_tests.txt`
- `results/phase0_gate.json`
- `results/test_output.txt`
- `results/tests.xml`
- `src/model/__init__.py`
- `src/moe/grouped.py`
- `src/moe/router.py`
- `src/training/engine.py`
- `src/training/moe_optimizer.py`
- `tests/test_moe_runtime.py`
- `tools/archive_prompt.py`
- `tools/moe_benchmark.py`
- `tools/moe_parity.py`
- `tools/moe_reference_profile.py`
- `tools/moe_report.py`
- `results/prompt-1_archive_manifest.json`
- `results/prompt-1_archive_verification.json`
- `results/moe_archive_output.txt` (initial sealing log; final refresh receipt is authoritative)

# Frozen repository corpus and final-data transition preflight

All preparation gates passed for `research_v2_repository_transition_2026_10_08_r2`. No new training campaign has started. The four canonical 100,007,936-token model/full-state pairs are hash-unchanged. [Readiness receipt](../results/research_v2_real/transition_readiness_r2.json) records the source, tokenizer, corpus, configuration and checkpoint identities.

## Corpus and recipe

The [reviewed source plan](../configs/research_v2_real/source_plan_r4.json) contains 44 pinned repository families: 26 training, 18 held out, with no family intersection. Committed license/notice identities, conservative file/subtree exclusions and source attribution are recorded per source. General reasoning is original Open Logic and Lean textbook/proof material; Open Logic uses [CC-BY-4.0](https://github.com/OpenLogicProject/OpenLogic), with creator/source/license attribution and transformations recorded in the plan. Raw source and large artifacts remain local.

Global deduplication reduced 133,996 exported files to 127,661. The final version holds another 2,447 low-information/repetitive files and corrects 23 `.h` metadata labels to C++. It retains 125,214 documents, **201,149,172 training tokens** and **30,774,317 validation tokens**. Counts include repository/file headers and packing markers. [Finalization](../results/research_v2_real/corpus_transition_finalization_r2.json) and [manifest](../results/research_v2_real/corpus_transition_manifest_r2.json) preserve exclusions and physical-shard hashes.

The natural retained training recipe is general 0.665%; technical 9.346%; code 86.975%; context 3.014%. No oversampling or synthetic development mixture weights drive the packed stream. Resolved phase configurations record these measured proportions. General reasoning is a narrow 0.665% logic/mathematics subset; this does not establish broad knowledge coverage.

FIM reconstruction and frozen-tokenizer roundtrips passed on all upstream retained documents. The final subset preserves source text and FIM structure, with 42,536 transformed documents among 106,341 eligible documents. Existing literal markers use the unchanged 32,768-token vocabulary. Document-isolated attention, boundary-target masking and Ngram history resets follow the frozen packing policy. CPU cursor resume matches exactly in both splits, including epoch transitions. The [measurement report](../results/research_v2_real/corpus_transition_measurement_r2.json) records the recipe and coverage.

| Code language | Train tokens | Validation tokens | Sampled held-out documents | Predictions |
|---|---:|---:|---:|---:|
| Python | 11,770,230 | 2,653,646 | 64 | 43,107 |
| JavaScript | 5,197,969 | 741,921 | 64 | 34,737 |
| TypeScript | 2,164,670 | 1,471,531 | 64 | 47,566 |
| C | 2,063,554 | 98,513 | 64 | 37,479 |
| C++ | 14,325,164 | 178,341 | 64 | 51,225 |
| Rust | 23,807,994 | 180,572 | 64 | 48,193 |
| Java | 25,374,135 | 2,084,677 | 64 | 50,243 |
| Go | 38,029,106 | 243,172 | 64 | 45,200 |
| C# | 44,676,208 | 13,344,870 | 64 | 36,013 |
| SQL | 4,616,525 | 347,586 | 28 | 24,364 |
| Bash | 769,221 | 188,348 | 64 | 39,816 |
| HTML/CSS | 952,770 | 394,093 | 64 | 37,474 |

All required language strata exceed 16 documents and 8,192 predictions. General has 24 sampled eligible documents / 23,144 predictions; technical has 64 / 55,366. The extra CMake code stratum has only two eligible documents and is not a supported language-quality claim. Model NLL uses the existing grouped JS/TS and C/C++ strata; this coverage table checks individual languages separately.

## Contamination and functional evidence

Original exports, transformed r1 and final r2 all have zero bounded lexical matches against pinned HumanEval (164), MBPP (974) and the project fixture (25). [Final screen](../results/research_v2_real/benchmark_overlap_transition_r2.json) records hashes and omitted short patterns. This checks exact casefolded lexical sequences, not semantic overlap, shared upstream ancestry or pretraining history.

Fresh canonical GPU generation produced 700 completions across four models, 25 elementary tasks and seven temperatures. Generation-time checkpoint hashes are verified in the [restricted WASI functional report](../results/research_v2_real/functional_canonical_gpu_r1.json). Greedy results are Dense 2/25, Sparse and both Ngram variants 0/25; neither correct Dense completion is free of the repetition heuristic. This is a small fixed-case completion probe, not pass@k, repository repair or agent capability. It uses BF16/TF32, seed 42, up to 128 new tokens, greedy at temperature 0 and top-k 40 / top-p 0.95 otherwise. The new protocol is distinct from historical saved-generation results.

## Transition and GPU gate

The proposed phase retains weights, Adam moments/steps and Python/NumPy/Torch/CUDA RNG, then resets the new-corpus cursor and phase-relative LR schedule. It adds 149,995,520 tokens in 18,310 updates of 8,192 tokens, reaching 250,003,456 total. LR warms up for 500 updates to 0.0003, then uses cosine decay to a 0.1 minimum ratio. Matrix decay excludes vectors/biases and Ngram tables; activation checkpointing is enabled. The tokenizer, architecture, context 1024 and microbatch 8 / accumulation 1 stay fixed. The phase covers 0.746 of one training pass, with no repeated full epoch. Global checkpoint/log/evaluation step cadence retains the inherited counter. [Initial-state receipt](../results/research_v2_real/transition_initial_states_r1.json) records every parent and new-state hash.

The RTX 5070 preflight uses strict deterministic algorithms, BF16 and TF32. Each model passes an actual-engine single update, exact equality with the custom first update, three warmup + six measured disposable updates, and an exact six-update interrupted replay. Model, Adam, scheduler, RNG, cursor and phase counters match; model/optimizer tensors are finite. Canonical and initial checkpoint hashes remain unchanged. [Completed preflight](../results/research_v2_real/transition_cuda_preflight_r2.json) includes routes, memory gate/bucket diagnostics, timing and ablations.

| Model | Mixed NLL | Code NLL | General NLL | Technical NLL | Train tokens/s | Peak allocated GiB | Exact resume |
|---|---:|---:|---:|---:|---:|---:|---|
| dense75_ref | 2.672521 | 2.514829 | 3.186912 | 3.763069 | 32,758 | 1.431 | passed |
| sparse75 | 2.651522 | 2.490495 | 3.163370 | 3.770306 | 26,194 | 1.678 | passed |
| sparse75_ngram10m | 2.656593 | 2.507606 | 3.170879 | 3.764524 | 25,618 | 1.814 | passed |
| sparse75_ngram25m | 2.632594 | 2.485663 | 3.177589 | 3.719481 | 25,861 | 1.978 | passed |

These NLLs evaluate inherited weights on the newly frozen corpus before updates: 128 mixed batches, plus up to 64 independently hashed held-out document windows per stratum. Categories cannot reconstruct the separately sampled mixed mean. Six timed updates are a short throughput screen, not sustained campaign throughput or a general speedup claim. Peak allocated memory covers the disposable training updates; reserved memory and checkpoint/evaluation times are recorded separately.

- sparse75_ngram10m: mixed NLL 2.656593, same-checkpoint residual off 2.682760.
- sparse75_ngram25m: mixed NLL 2.632594, same-checkpoint residual off 2.673583.

The first CUDA attempt stopped at a gate-histogram diagnostic because CUDA float `histc` rejects strict determinism. Histogram counting now uses the deterministic CPU kernel; a CUDA regression reproduces that condition. All four models passed a distinct r2 attempt. Failed/disposable artifacts are retained, and [the failure receipt](../results/research_v2_real/transition_cuda_preflight_failure_r1.json) preserves the cause. The upstream r1 pipeline also has an explicit [source-provenance supplement](../results/research_v2_real/corpus_transition_provenance_r1.json), distinguishing its loaded interpreter snapshot from its completion-time on-disk script hash.

## Storage and validation

Active training data uses 0.585 GiB within a 6 GiB budget. The measured volume has 39.65 GiB free; after reserving four rolling and four promoted full states, four model-only gate weights and one atomic replacement, 30.23 GiB remains, above the 20 GiB reserve. Canonical/history and all preparation/disposable states are retained. [Storage receipt](../results/research_v2_real/transition_storage_r2.json) measures explicit artifact scopes without scanning unrelated project files.

The final regression passes 118 tests, including GPU checks. Large datasets, caches, source payloads, checkpoints and generation text are outside ordinary Git. Preparation readiness is complete; a longer training phase and subsequent capability conclusions are outside this run.

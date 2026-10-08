# Repository corpus and data transition preparation

The expanded corpus is frozen and all preparation gates passed. No new training campaign has started. The [completed preflight record](60_frozen_repository_transition_preflight.md) gives measured mixture, coverage, storage and CUDA resume results; the four immutable 100,007,936-token model/full-state checkpoints remain the common starting point.

## Sources and preprocessing

The [final source plan](../configs/research_v2_real/source_plan_r4.json) records 44 pinned repositories, committed license/notice hashes, GitHub fork metadata, explicit repository-family split assignments and conservative source exclusions. Root licenses do not override file exceptions. Nested separately licensed subtrees, ambiguous copied-content headers, unsupported SPDX expressions and unreviewed licenses are omitted. This deliberately sacrifices some usable code; it is a review of included subsets, not a license certification of entire projects.

General reasoning comes from original textbook/proof sources. [Open Logic](https://github.com/OpenLogicProject/OpenLogic) uses CC-BY-4.0; creator/source/license attribution and the header/packing modifications are recorded in the plan. Lean textbook sources retain their separately recorded Apache-2.0 terms. Raw source, downloads, caches, shards, checkpoints and generation text remain local.

`tools.prepare_repository_corpus` verifies committed revision and notice identities, exports bare Git blobs through one batch reader, deduplicates globally before splits/FIM, checks exact reconstruction and frozen-tokenizer roundtrips, and packs document-isolated shards. The final configuration is [repo_transition_r2.yaml](../configs/research_v2_real/repo_transition_r2.yaml). Token/language/role counts, held-out sufficiency, storage and repeat exposure are measured in the completed preflight record.

`tools.screen_benchmark_overlap` checks pinned HumanEval, MBPP and project-fixture registries using exact casefolded lexical sequences. References below 32 tokens and prompts below 12 tokens are omitted. It cannot establish semantic or historical contamination freedom. `tools.finalize_repository_corpus` creates a new version after holding matching files and extreme low-information/repetitive material; frozen versions are never edited.

## Explicit phase transition

`tools.prepare_data_transition` verifies canonical hashes and model-only/full-state equality, retains Adam moments/steps and Python/NumPy/Torch/CUDA RNG, and resets the new-corpus cursor to 0. The proposed phase has 18,310 updates of 8,192 tokens, adding 149,995,520 tokens to reach 250,003,456 total. Its phase-relative scheduler uses 500 warmup updates, cosine decay, peak LR 0.0003 and minimum ratio 0.1. Matrix decay excludes vectors/biases and Ngram tables. Activation checkpointing is enabled; architecture, tokenizer and effective batch stay fixed.

`tools.transition_cuda_preflight` passes all four models on the final corpus in a distinct r2 attempt. It evaluates inherited weights, measures disposable updates, saves atomically and compares interrupted versus uninterrupted model/Adam/scheduler/RNG/cursor state. The new phase explicitly reports inherited and new-data token counters, uses a scoped active-data budget and reserves 20 GiB before checkpoint replacement.

## Evaluation and validation status

`tools.generate_canonical_python` checks immutable weight hashes before/after sampling and uses batches of 4 with a final-position vocabulary projection. CPU, CUDA FP32 and CUDA BF16 checks verify causal-prefix padding and final-position logits within declared numerical tolerances. Generation records carry checkpoint hashes; functional scoring uses the existing restricted WASI runtime. This remains a small coding probe, not a coding-agent capability result.

The final source revision passes 118 tests, including GPU tests. Expanded preprocessing, three bounded overlap screens, final-data CUDA resume, functional scoring and measured corpus/storage reports are complete; see the [final record](60_frozen_repository_transition_preflight.md). The earlier candidate plan/config remains a historical preparation recipe.

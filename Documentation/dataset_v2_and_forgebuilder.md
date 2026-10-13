# Dataset-v2 and ForgeBuilder infrastructure

Dataset V2 construction is in progress. The infrastructure defines versioned records, immutable content-addressed storage,
deterministic manifests and splits, benchmark quarantine metadata, sequential
resumable shards, model-batched generation interfaces, a transactional work
queue, critic caching, objective verification results, repository-task gates,
and linked student-failure/recovery records.

The foundation builder reuses reviewed pinned repository sources, preserves the existing held-out splits, and creates canonical file, FIM and repository views. Reuse and additional views do not count as new unique source material. A separate local Flash-Next runtime reviews source-grounded pilot candidates. Student training remains disabled pending a frozen two-arm comparison protocol, verified data and separate launch authorization. No architecture winner has been frozen.

## Record and storage rules

Records use schema version 1 and preserve unknown JSON fields so modality- and
source-specific metadata can be added without silently dropping it. Unknown
license metadata is represented as `null`; provenance is required. Stable IDs
are derived from source identity, content hash, task family, transformation,
and parent sample IDs rather than timestamps or mutable review fields.

Use `data/dataset_v2/` for the private store, with separate locations for
references, canonical objects, repository snapshots, tasks, trajectories,
student failures, recoveries, verified/rejected records, manifests, indexes,
shards, and cache. The content-addressed store writes immutable SHA-256 objects.
For source material that cannot be redistributed, keep the exact source URL,
revision, license, expected digest, and extraction recipe instead of retaining
the payload.

Manifests are canonical JSONL sorted by stable sample ID. Repository split
assignment is deterministic and repository-level so files from one repository
cannot leak across train, validation, and test. The lexical eval screen records
its method and matches; a clear lexical result is not a universal contamination
guarantee. Frozen evaluation material must remain quarantined by explicit
metadata as well as screening.

Shards are sequential JSONL files with digests and an atomic manifest. A
`.partial` shard is validated and resumed after interruption. Keep the index
and hot cache on fast storage where practical; consolidate records into large
shards instead of creating one file per sample.

The working storage plan stays within the roughly 1.6 TB practical project
budget. Keep canonical high-value sources, manifests, current repository
snapshots, current shards, and hot caches. Avoid retaining every rejected
candidate, duplicate checkout, container image, and derived view indefinitely.
The large Flash Ngram table should use NVMe when available rather than the
archive HDD.

## ForgeBuilder quality gates

A repository task binds an exact base commit, fixed commit, environment hash,
source provenance, fail-to-pass tests, and pass-to-pass regression tests. The
fail-to-pass tests must fail before the fix and pass after it; regression tests
must pass at both points. Candidate acceptance requires both test groups to
pass, an independent critic PASS, and an inspected diff. A teacher judgment by
itself cannot establish correctness. The exact-line patch leakage check is a
screen only, not a proof of semantic isolation.

Repository checkout and isolated execution are represented by interfaces; no
general-purpose untrusted-code runner is enabled by this scaffold. An eventual
adapter must provide an isolated environment, explicit resource limits,
captured tool evidence, and deterministic test identity before executing
candidate patches.

## Generation and critic flow

The scheduler groups requests by generator model, task family, and context
bucket, then sorts them deterministically. This supports one model load per
batch. The SQLite task queue uses leases so expired in-flight tasks return to
pending. The critic cache key binds task, context, candidate, patch, objective
evidence, model/config, and prompt version.

The profile file names the full Flash-Next critic and provides 32K, 64K, 128K,
and 256K context targets. The current local runtime uses 32K; larger profiles remain unmeasured. Critic review is
reserved for valuable synthetic/post-training samples; raw foundation code
uses provenance, license, quality, deduplication, parser, and quarantine
filters first. If Flash generated a candidate, critique must happen in a fresh
context after deterministic verification.

Bonsai 2 27B PQ2_0 and Ornith 1.5 35B-A3B Q4_K_M have completed small local function qualifications at 32K. Their intended bulk-coding and repository/SWE roles still require broader task coverage, resource measurements and context scaling before production routing. Stored GGUF size alone is not a rejection criterion. Flash-Next IQ3_XXS reviews candidates through the isolated runtime described below.

## Student errors and recovery

Failure records preserve state, evidence, hypothesis, environment, model
version, recurrence, and confidence where known. Known-bad actions are marked
with zero target weight while remaining available as context. A recovery record
links to its failure and is accepted only when its objective verification
passes. These records are compatible with later retrieval and controlled
adapter experiments; they do not trigger online weight updates.

## Current validation scope

The [foundation receipt](../results/dataset_v2/foundation_v1.json) records 124,467 canonical files and 71,039 derived views from 43 reviewed repository families. It excludes 712 records outside the reviewed source plan, quarantines 34 lexical evaluation overlaps and one Python parse failure, and registers 2,131 distinct evaluation negatives. The actual mixture remains uneven: the issues/PR/tests category currently contains tests, and broad multilingual/general text and real issue/PR patch coverage are incomplete. These counts do not establish training readiness.

The [qualification snapshot](../results/dataset_v2/source_pilot_qualification_v1.json) contains 4,165 prepared candidates against a 5,000 target, with 58 reviewed/accepted records in the saved snapshot. Additional reviews run separately and do not change that historical snapshot. The source extractor exhausted its eligible two-function excerpts; remaining coverage must come from further verified tasks rather than duplicated candidates.

Tests use tiny synthetic fixtures only. They exercise schema checks, stable
IDs, deterministic manifests and splits, quarantine screening, license
metadata, STOP/CONTINUE task metadata, critic caching, verifier results,
failure/recovery links, rejection reasons, resumable shards, queue recovery,
model-centric batching, and manifest round trips. No frozen benchmark prompt
is used as a test fixture.

The first pilot targets 5,000 source-grounded candidates covering excerpt-grounded reasoning, missing-symbol checks, constructed recovery and paired STOP/CONTINUE extraction. Exact source/AST checks run before independent critic review. These checks establish extraction and evidence support, not executable behavior or SWE patch correctness. A separate six-solution code-synthesis pool has passed finite WASI checks and independent critique. Actual tool/terminal trajectories, real fail-to-pass repository patches and student-on-policy recovery remain pending; this pilot alone does not satisfy the complete dataset coverage requirement. Both Ngram arms are retained. The
configuration records the required per-family quality, correction, rejection,
stop/scope, recovery, timing, and storage metrics. Generator and compute-target
files for remote compute remain planning profiles. No remote inference services or TPU/TRC allocation are used.

The [pinned local-model inventory](../configs/dataset_v2/local_models.json) selects the publisher's Bonsai 2 27B PQ2_0 and Ornith 1.5 35B-A3B Q4_K_M files for download and measurement. Bonsai requires PrismML's activation-transform-capable runtime; the existing local PrismML llama.cpp fork is available. The downloader uses resumable, checked HTTP byte ranges and verifies publisher file sizes and SHA256 before exposing final model filenames. Both files passed publisher size/SHA256 verification. Ornith runs from D:/models/dataset_v2/ornith15_35b after relocation from the HDD; its local compatibility and small-function qualification are measured below. No vision projector download is needed for the current text-only generator tasks.

Bonsai's [local qualification](../results/dataset_v2/bonsai_generator_qualification_v1.json) and Ornith's [local qualification](../results/dataset_v2/ornith_generator_qualification_v1.json) each passed three project-specific tasks and 15 WASI cases. Median decoded throughput was 64.87 and 49.63 tokens/s respectively, including thinking tokens. These are small functional qualifications, not general coding benchmarks or controlled model-throughput comparisons.

The [generated-code snapshot](../results/dataset_v2/generated_pilot_qualification_v1.json) contains six solutions that passed independent WASI re-verification and fresh-context Flash critique. Raw responses, transformations and model identities are retained privately. Finite tests do not prove universal correctness or repository patch quality. Generated output licenses remain unknown; the model license is not assigned to outputs. The snapshot is not a frozen training dataset. Reviews span two expert-cache reserve settings, so their timings do not establish a controlled performance comparison.

## Local 32K Flash runtime

`python -m tools.dataset_v2_strata start` copies the installed IQ3_XXS configuration and shared settings into `.local/dataset_v2/strata32k/`, then invokes the existing Strata server and engine. Model arguments inherit the installation except `--max-context 32768` and `--vram-reserve-mib 2560`. The reserve allows roughly 2 GiB free after extra engine buffers; actual free memory varies with concurrent applications. The [headroom receipt](../results/dataset_v2/teacher_runtime_headroom_v2.json) records 2,374 MiB free at measurement and 2,301 MiB at engine startup. The initial near-full allocation remains historical evidence. Host, port and logs are isolated at `127.0.0.1:8096`. The main Strata configuration and shared settings are never edited. The first model shard stays on D: and the existing second-shard symlink resolves to C:.

`python -m tools.dataset_v2_strata fingerprint` hashes the actual model shards, converted packs, tokenizer, MTP, control vector and vision assets before critic review. The critic cache binds those identities, task/evidence/candidate, runtime configuration and review version. Each review starts a fresh chat context. Incomplete or malformed critic output leaves the review pending; it does not make the candidate functionally incorrect.

`python -m tools.build_dataset_v2` constructs the foundation manifests. `python -m tools.finalize_dataset_v2_foundation` sorts them by stable sample ID with bounded memory and a recovery journal. `python -m tools.dataset_v2_pilot prepare` builds the pilot candidates; `review` resumes independent local review from durable receipts and cache entries. The campaign worker sequences these stages without launching student training. Create `data/dataset_v2/source_pilot_v1/STOP` to stop review after its current request; `.local/dataset_v2/STOP` prevents later campaign stages.

## Planned two-arm continuation

The [500M review](64_matched_500m_results.md) supports retaining +10M and +25M rather than declaring a final architecture winner. The [disabled comparison plan](../configs/dataset_v2/comparison_plan.json) starts at 499,998,720 tokens and requires matched evaluations at 625,000,448 and 750,002,176 tokens. Dataset V2 construction has started; student training and both continuation gates remain unstarted. Architecture, tokenizer and student context remain fixed; the eventual shared data stream, scheduler transition and exact-resume policy must be validated before launch.


## Read-only continuation entry checks

`python -m tools.inspect_dataset_v2_entries` checks the two pinned full states on CPU using memory-mapped checkpoint reads and a model constructed on the meta device. The [entry inspection](../results/dataset_v2/entry_preflight_v1.json) verifies both states at 499,998,720 tokens / 61,035 updates, model tensor shapes and dtypes, all 221 optimizer parameter bindings per arm, moment shapes, saved RNG presence, scheduler counters and learning-rate floor. It also verifies tokenizer/source-manifest hashes and identical next historical source batches from cursor 390,616. The inspection does not update weights or write replacement checkpoints.

The saved configuration is read as JSON to preserve its numeric types and original fingerprint. The generic YAML loader interprets an exponent literal differently, so it is not used to reconstruct these JSON checkpoint identities. Historical scheduler counters are retained: phase step 48,827 and learning rate 0.00003. The old config's 250M target is historical; it does not define the planned comparison endpoint.

This is structural entry validation, not proof of interrupted-versus-continuous numerical equivalence. CUDA RNG restoration, the Dataset V2 scheduler/data-cursor transition, final packing/mixture/order, peak storage budget and explicit student-training authorization remain open launch gates. Current disk free space clears the existing 20 GiB reserve, but that observation does not establish capacity for the eventual full dataset and atomic checkpoint peak.

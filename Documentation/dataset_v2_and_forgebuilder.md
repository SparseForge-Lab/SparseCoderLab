# Dataset-v2 and ForgeBuilder infrastructure

This is the initial, offline-capable infrastructure for a future Dataset-v2
pilot. It defines versioned records, immutable content-addressed storage,
deterministic manifests and splits, benchmark quarantine metadata, sequential
resumable shards, model-batched generation interfaces, a transactional work
queue, critic caching, objective verification results, repository-task gates,
and linked student-failure/recovery records.

The scaffolding is deliberately not a data-generation run. It does not load
Bonsai, Ornith, or Flash, fetch external repositories, execute candidate code,
or launch Dataset-v2 training. Those actions remain out of scope until the
matched 500M comparison is complete and a winner is frozen.

## Record and storage rules

Records use schema version 1 and preserve unknown JSON fields so modality- and
source-specific metadata can be added without silently dropping it. Unknown
license metadata is represented as `null`; provenance is required. Stable IDs
are derived from source identity, content hash, task family, transformation,
and parent sample IDs rather than timestamps or mutable review fields.

Use `data/dataset_v2/` for the eventual store, with separate locations for
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
and 256K context targets. No default profile is selected until throughput,
memory, acceptance, and correction rates are measured. Critic review is
reserved for valuable synthetic/post-training samples; raw foundation code
uses provenance, license, quality, deduplication, parser, and quarantine
filters first. If Flash generated a candidate, critique must happen in a fresh
context after deterministic verification.

Generator profiles are configuration only. Bonsai 2 27B remains a bulk
generation candidate whose quantization and output quality need measurement.
Ornith 1.5 35B-A3B remains an agentic/SWE candidate; stored GGUF size alone is
not used as a rejection criterion. Load time, prompt/decode speed, RAM, VRAM,
context scaling, and task quality must be measured before routing production
batches. No local model was loaded for this infrastructure work.

## Student errors and recovery

Failure records preserve state, evidence, hypothesis, environment, model
version, recurrence, and confidence where known. Known-bad actions are marked
with zero target weight while remaining available as context. A recovery record
links to its failure and is accepted only when its objective verification
passes. These records are compatible with later retrieval and controlled
adapter experiments; they do not trigger online weight updates.

## Current validation scope

Tests use tiny synthetic fixtures only. They exercise schema checks, stable
IDs, deterministic manifests and splits, quarantine screening, license
metadata, STOP/CONTINUE task metadata, critic caching, verifier results,
failure/recovery links, rejection reasons, resumable shards, queue recovery,
model-centric batching, and manifest round trips. No frozen benchmark prompt
is used as a test fixture.

The 5K–10K pilot configuration is present but disabled. Its gate requires all
three 500M arms, the matched evaluation, and a frozen winner first. The
configuration records the required per-family quality, correction, rejection,
stop/scope, recovery, timing, and storage metrics. Generator and compute-target
files are planning profiles only; they do not load local teacher models, call
remote services, or assume TPU/TRC permission.

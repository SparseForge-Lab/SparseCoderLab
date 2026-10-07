# Repository data pipeline

The next corpus version will keep the four models at the same 100M transition point while changing the data recipe. The existing corpus already contains real code and documents. The aim here is to improve repository coverage, preserve source/test/documentation relationships, add suitable FIM examples, and strengthen filtering and evaluation.

## First preprocessing test

I started with two small repositories that have explicit root licenses: [Flask](https://github.com/pallets/flask/blob/d086db856be187255b8ec61ef409357393020f32/LICENSE.txt), under BSD-3-Clause, and [Requests](https://github.com/psf/requests/blob/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60/LICENSE), under Apache-2.0. This is a pipeline fixture rather than the planned training corpus. Their source files and token shards remain outside Git.

| Repository | Pinned commit | Exported files | Export bytes | Documents after dedup | Packed tokens | Split |
|---|---|---:|---:|---:|---:|---|
| pallets/flask | `d086db856be187255b8ec61ef409357393020f32` | 218 | 1,350,825 | 214 | 302,798 | Validation |
| psf/requests | `611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60` | 90 | 696,848 | 89 | 164,187 | Train |

The importer reads Git blobs from the recorded commit rather than working-copy bytes. Each record carries a stable repository ID, canonical URL, commit, relative file path, content hash, Git blob ID, root-license hash, language, and file role. Documentation, tests, build files and configuration are retained alongside implementation files. The root-license assertion does not resolve every possible per-file exception; a larger corpus still needs that review.

Before deduplication, the fixture contained 72 implementation files, 82 test files, 125 documentation files, 8 build files and 21 configuration files. Two generated-header files and two symlink/submodule entries were excluded; unsupported file types were also excluded. The secret-pattern quarantine produced no entries for these two repositories. Its metadata contains paths and reasons, never matched secret values.

Global exact/normalized body hashes removed five duplicate files, even when repository headers differed. The split hashes seed 42 and the repository ID, keeping all files from each repository together. The smoke configuration uses a 20% hash threshold to exercise both splits with two repositories; its realized validation share is much larger and is not a target for the full corpus. Token packing uses BOS/body/EOS/document separators and bounded uint16 shards. Causal attention can cross document boundaries within one split.

All 308 imported records round-tripped exactly through the unchanged 32,768-token tokenizer. Python used 0.261 tokens per character in this fixture and reStructuredText used 0.258. These estimates describe two Python projects, not broad language coverage. The tokenizer SHA-256 remains `832daf453fc64e28f50df00ac2e458718576bb26e3f16e737f18b120ae4e309a`.

The [smoke receipt](../results/research_v2_real/repository_smoke.json) records source and license hashes, exclusions, file roles, tokenizer measurements and split membership. The [corpus manifest](../results/research_v2_real/corpus_smoke_manifest.json) records token counts, source revisions and shard hashes. Eight tests passed across provenance, repository splitting and the existing corpus filters. No model was trained on this fixture.

## Sources for a larger corpus

The [source catalogue](../data/research_v2_sources.yaml) distinguishes the measured pilot from candidates that still need approval or additional metadata. An expanded [repository inventory](../results/research_v2_real/source_inventory_pilot.json) now covers 13 pinned projects and 5,895 files across Python, JavaScript, TypeScript, C, C++, Rust, Java, Go, C#, SQL, shell and HTML/CSS. It records 32,843,880 export bytes before global deduplication; file counts are not token proportions. This remains a preprocessing pilot with licensing exceptions, fork review, contamination checks and the final mixture still open.

| Source | Useful material | Current status |
|---|---|---|
| Curated, pinned repository snapshots | Source, tests, docs and build configuration with file relationships | Pilot ingestion works; expand through an explicit reviewed repository list |
| [CodeSearchNet](https://github.com/github/CodeSearchNet) | Function/comment pairs in six languages, with repository identifiers | Candidate for an auxiliary code/docs pool; filter through the separate per-repository license maps |
| [The Stack Dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup/blob/main/README.md) | Broad multilingual code coverage with provenance | Held out: access is gated, source-license conditions apply, and the usable-version/removal policy must be satisfied |
| Existing CodeParrot snapshot | Previously downloaded, license-annotated files with repository paths | Retain as historical evidence; any reuse needs an explicit new-version audit and must be reported as reused data |
| Existing FineWeb-Edu snapshot | Educational web text | Held out from the new recipe pending the individual-content provenance/terms decision; collection licensing does not identify each page's license |

CodeSearchNet's tooling license does not replace the licenses of its source examples. Its maintainers provide [per-repository license maps](https://github.com/github/CodeSearchNet/blob/106e827405c968597da938f6b373d30183918869/resources/README.md). Function snippets alone also do not provide complete repository context.

The Stack requires compliance with the original licenses, an update policy for validated removals, and additional conditions when redistributing a copy. No access terms were accepted and no files were downloaded. Its headline size is not a usable-token estimate for this project.

## Before the first training gate

The working mixture remains a research hypothesis: 60% code/repositories, 15% technical documentation, 15% structured engineering context and 10% general reasoning. Final proportions will follow usable-token counts and source review. The corpus must cover the intended programming languages, isolate repositories and fork families across splits, and keep benchmark tasks out of training.

Exact and near-duplicate filtering now run before FIM and splitting. MinHash with 64 permutations and 16 four-row bands proposes matches from lexical five-token shingles; exact shingle Jaccard of at least 0.85 confirms a removal. Files with fewer than 64 distinct shingles receive exact filtering only. The two-repository audit compared 2,190 retained-file pairs and found no missed pair above the threshold. It removed five exact duplicates and no near duplicates. This validates the fixture, not corpus-wide recall or semantic deduplication. Reviewed fork-family declarations can group repositories before splitting; file similarity alone is not treated as proof of a fork.

FIM uses the existing prefix/suffix/middle delimiters (token IDs 10, 11 and 12). It changes neither tokenizer bytes nor embedding sizes. Selection and span boundaries are deterministic from the seed and source identity, and only suitable code files are eligible. The fixture converted 59 of 122 eligible files, or 48.4%, at a configured 40% selection probability. The small-sample realized fraction is recorded rather than substituted with the target. Every transformed sample reconstructed its original source exactly and round-tripped through the tokenizer.

The combined [dedup/FIM/packing receipt](../results/research_v2_real/pipeline_fim_smoke.json) records 164,253 train tokens and 302,981 validation tokens. Both splits passed an exact packed-stream cursor resume check on CPU. This checks data continuation, not optimizer or model checkpoint continuation. Original source hashes, FIM status and repository-family metadata survive into packed-document sidecars.

The first [dependency-context fixture](../results/research_v2_real/neighbors_smoke.json) found 298 local import edges in the Flask/Requests records and constructed 82 bounded multi-file samples, containing 267,374 tokens. It retains each member's file/license hashes and repository revision. Python links use AST imports; other supported languages use conservative text matches. A link from a test file to implementation is not evidence that the test was executed. This lane deliberately reuses source files, so its token count is separate from unique corpus tokens. Broader dependency resolution and contamination checks remain open. Neither preprocessing receipt certifies long-run readiness.

The first training gate is approximately 250M total tokens: 100,007,936 inherited tokens plus about 150M tokens from the new phase. It will require a frozen source inventory, measured train/validation and language totals, a documented optimizer/LR transition, checkpoint/resume tests, functional coding evaluation, and a runtime benchmark with all four actual checkpoint weights. No larger run is implied by this preprocessing result.

Reproduce the pilot from clean checkouts at the commits above:

```powershell
python -m tools.import_local_repo --repo data/research_v2_real/raw/repos/flask --source-id pallets/flask --source-url https://github.com/pallets/flask --license BSD-3-Clause --output data/research_v2_real/exports/flask.jsonl
python -m tools.import_local_repo --repo data/research_v2_real/raw/repos/requests --source-id psf/requests --source-url https://github.com/psf/requests --license Apache-2.0 --output data/research_v2_real/exports/requests.jsonl
python -m tools.repository_data_smoke
python -m tools.repository_dedup --audit-all-pairs
python -m tools.repository_pipeline_smoke
python -m tools.repository_neighbors
python -m pytest -q tests/test_repo_provenance.py tests/test_research_data.py tests/test_repository_dedup.py tests/test_repository_fim.py tests/test_repository_neighbors.py
```

Exports and frozen shard directories refuse overwrite. For a rerun, use a new version and output paths.

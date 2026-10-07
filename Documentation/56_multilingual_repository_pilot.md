# Multilingual repository pilot

I used 13 small, pinned repositories to test source, tests, documentation and build/configuration files through the same preprocessing path. The frozen pilot version is `research_v2_repository_pilot_multilang_2026_10_07_r5`.

The import inventory contained 5,895 files and 32,843,880 JSONL bytes. Global exact/near filtering retained 5,526 files. Packing produced 7,151,137 training tokens and 595,542 validation tokens. These are measured pilot counts; the final training mixture and source inventory remain under review.

| Repository | License assertion | Packed tokens | Split |
|---|---|---:|---|
| [bats-core/bats-core](https://github.com/bats-core/bats-core/tree/8c8ec0b1d552f279853b76b9d80b3b6fcb1d7a4d) | MIT | 173,323 | train |
| [spf13/cobra](https://github.com/spf13/cobra/tree/adbc8813901bba65827259daa8e22ff94ec1f30e) | Apache-2.0 | 191,532 | train |
| [DapperLib/Dapper](https://github.com/DapperLib/Dapper/tree/eb47546a408bbf6c178bda4f04dc205bc51ffbfe) | Apache-2.0 | 292,774 | val |
| [expressjs/express](https://github.com/expressjs/express/tree/9efc29e280018dafc1f0617f3a3d28f4e463b7be) | MIT | 204,804 | train |
| [pallets/flask](https://github.com/pallets/flask/tree/d086db856be187255b8ec61ef409357393020f32) | BSD-3-Clause | 302,768 | val |
| [fmtlib/fmt](https://github.com/fmtlib/fmt/tree/c4927e6086c2aa5bae0f5c2bce791c369c055197) | MIT | 933,309 | train |
| [google/gson](https://github.com/google/gson/tree/845664ba1c307e6c1910d07cfed2f622e0ad8df1) | Apache-2.0 | 556,142 | train |
| [benhoyt/inih](https://github.com/benhoyt/inih/tree/2bbdec4a366c8c39746ee0982e7ca0febbb044b6) | BSD-3-Clause | 22,964 | train |
| [necolas/normalize.css](https://github.com/necolas/normalize.css/tree/fc091cce1534909334c1911709a39c22d406977b) | MIT | 11,379 | train |
| [psf/requests](https://github.com/psf/requests/tree/611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60) | Apache-2.0 | 164,253 | train |
| [serde-rs/serde](https://github.com/serde-rs/serde/tree/6693a89cca77e0151437da1c7f890090b9ebf04c) | MIT | 345,260 | train |
| [sqlfluff/sqlfluff](https://github.com/sqlfluff/sqlfluff/tree/67f23c9a1c5fbaf72583cf903102336bfc2b6bc2) | MIT | 3,050,617 | train |
| [colinhacks/zod](https://github.com/colinhacks/zod/tree/0b216ef674e297ebe41d8bf902262e56f8755822) | MIT | 1,497,554 | train |

## Token coverage

| Language | Train tokens | Validation tokens |
|---|---:|---:|
| Bash | 126,470 | 78 |
| C | 10,641 | 0 |
| C# | 0 | 255,890 |
| C++ | 771,919 | 0 |
| CMake | 14,560 | 0 |
| Dockerfile | 1,706 | 0 |
| Go | 159,848 | 0 |
| HTML/CSS | 19,501 | 4,513 |
| JSON/YAML | 609,714 | 7,850 |
| Java | 518,578 | 0 |
| JavaScript | 157,461 | 0 |
| Makefile | 4,238 | 0 |
| Markdown | 782,416 | 17,727 |
| Python | 1,619,902 | 155,064 |
| Rust | 591,418 | 0 |
| SQL | 549,880 | 360 |
| TOML | 12,466 | 3,160 |
| Text | 28,856 | 16,339 |
| TypeScript | 1,130,682 | 0 |
| reStructuredText | 40,881 | 134,561 |

Language totals include tests and configuration where applicable. Most languages currently have no independent validation repository in this small inventory; language-specific quality comparisons need a broader held-out set.

| Training category | Tokens | Share |
|---|---:|---:|
| code | 5,900,402 | 82.5% |
| context | 384,422 | 5.4% |
| technical | 866,313 | 12.1% |

The pilot has no general-reasoning source. Its observed mixture does not implement the proposed 60/15/15/10 training recipe. Test/source/build context is useful material, but configuration and test files alone do not provide executable-agent trajectories.

FIM transformed 1,533 of 3,695 eligible code files (41.5%), using the unchanged vocabulary and existing delimiter IDs 10/11/12. All 5,526 retained records passed exact reconstruction and tokenizer roundtrips. Both splits passed physical-shard verification and exact packed-stream cursor continuation on CPU.

## Filtering and remaining work

Global deduplication runs before FIM and repository-family splitting. The [pipeline receipt](../results/research_v2_real/pipeline_multilang_pilot_v5.json) records removal counts and confirmed Jaccard matches. The exhaustive pair audit was limited to the earlier two-repository fixture; this larger pilot uses LSH proposals with exact similarity confirmation and may miss pairs.

Root-license files, selected revisions and hashes are recorded in the [source inventory](../results/research_v2_real/source_inventory_pilot.json). Conflicting SPDX/GNU headers, generated material, secret-pattern files and symlink/submodule entries are excluded. Full per-file licensing exceptions, fork-family review, benchmark contamination and broader validation coverage are still open. Source payloads and token shards stay outside Git.

A 150M-token exposure on the current 7,151,137-token training split would repeat it about 21.0 times. That is not 150M unique tokens. Further source expansion and a documented repeat/mix policy are needed before freezing a long-run recipe.

The [preprocessing provenance](../results/research_v2_real/preprocessing_provenance_v5.json) binds source/tool/configuration hashes to the [corpus manifest](../results/research_v2_real/corpus_multilang_pilot_manifest_v5.json). Sixty-three CPU regression tests passed. Model/optimizer resume, runtime measurements, checkpoint writes and functional coding evaluation remain separate training gates.

Reproduce from clean checkouts at the pinned revisions and the exported files in the inventory:

```powershell
python -m tools.repository_pipeline_smoke --config configs/research_v2_real/repo_multilang_pilot.yaml --report results/research_v2_real/pipeline_multilang_pilot_v5.json --exports data/research_v2_real/exports/bats-core.jsonl data/research_v2_real/exports/cobra.jsonl data/research_v2_real/exports/dapper.jsonl data/research_v2_real/exports/express.jsonl data/research_v2_real/exports/flask.jsonl data/research_v2_real/exports/fmt.jsonl data/research_v2_real/exports/gson.jsonl data/research_v2_real/exports/inih.jsonl data/research_v2_real/exports/normalize-css.jsonl data/research_v2_real/exports/requests.jsonl data/research_v2_real/exports/serde.jsonl data/research_v2_real/exports/sqlfluff.jsonl data/research_v2_real/exports/zod.jsonl
```

Frozen directories and receipts refuse overwrite. Reruns need new versioned paths.

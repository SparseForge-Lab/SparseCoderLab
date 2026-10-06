# Data and licenses

The initial corpus consists entirely of original CC0 synthetic engineering fixtures: about 2.77M train tokens and 0.143M held-out tokens. Nothing was scraped or downloaded from a dataset. Template repetition makes this unsuitable for measuring real coding quality; identifiers, values and complete documents are disjoint across splits, but templates intentionally recur.

Configured document mixture: code 55%, general 30%, technical 10%, context/tool tasks 5%. Documents have different lengths, so realized token proportions differ; `data/manifest.json` reports both. For real experiments choose token-balanced bounded source shards and inspect their actual proportions; do not infer token balance from document weights.

`tools.prepare_data` uses seeded generation or an explicit local JSONL, content SHA256 deduplication, content-hash train/val assignment, uint16 packed tokens and source/category/license metadata. Packing preserves EOS/document separators; attention can cross documents within one split. A restartable permutation over packed offsets preserves data order across architectures.

Commands:

```powershell
.\.venv\Scripts\python.exe -m tools.prepare_data
# Use a new config with a different shards/cache path for a new corpus.
.\.venv\Scripts\python.exe -m tools.prepare_data --config configs/my_corpus.yaml --input-jsonl data/approved.jsonl
```

Every imported record requires text, kind, language, source and an allowlisted license. `tools.import_local_repo` reads only tracked files from an explicitly specified licensed Git repository. A repository-level license assertion does not resolve per-file exceptions; review them before real training. `tools.stream_source` only accepts explicitly enabled HTTPS Hugging Face JSONL URLs with accepted terms, reads bounded lines/bytes and validates per-record code licenses. Disabled examples live in `data/sources.example.yaml`. Parquet/gzip sources need a separately reviewed streaming decoder. The Stack requires independently verified gated access; no access is requested automatically.

Default cache/data budget: 20 GiB. Default total project/checkpoint budget: 30 GiB. No giant dataset downloads. For real runs keep the tokenizer fixed, emit a new manifest, and never mix old and new shards in a resumed checkpoint.

Physical train/val shards contain at most `data.shard_tokens` tokens (default 262,144). Lazy memmaps read across shard boundaries without loading the corpus. Every shard has a manifest SHA256, verified before training/resume. `tools.shard_data` converts the development split files without changing their concatenated token order. The default cache holds about 10.8 MB of document JSONL plus small token shards and metadata.

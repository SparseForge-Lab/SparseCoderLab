# Research corpus — not prepared after review stop

The separate data/research_v1 and configs/research_v1 directories exist, with readiness notes. No real dataset content was downloaded, no research shards or tokenizer were created, and no research model configs were frozen. The 100M–250M usable-token target was not met. These are intentional omissions following the user's explicit slowdown stop/review condition, not a claim of completed corpus preparation. The original synthetic corpus/tokenizer and Phase 0 YAML files remain unchanged.

Only source cards and dataset API metadata were reviewed. Exact candidate revisions and a bounded file inventory are saved in results/research_source_review.json. This metadata is not a document corpus.

| Candidate | Primary evidence | Disposition |
|---|---|---|
| codeparrot/github-code | [Dataset card](https://huggingface.co/datasets/codeparrot/github-code/blob/b5661e6b17396364b2bcf8e68977b0d28e1ebd19/README.md) supplies code, repository, path, language and repository-license annotations | Candidate only; future ingestion must whitelist permissive per-record licenses and retain snapshot/file/row/repository/path provenance. Mixed dataset-level licensing cannot replace record filtering. No documents inspected or admitted. |
| FineWeb-Edu | [Dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md) declares ODC-By and CommonCrawl terms | Candidate only; retain collection license/terms and per-document URL/crawl metadata. Collection licensing is distinct from individual page ownership. No content admitted. |
| SmolLM Python-Edu | [Provider card](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/main/README.md) describes Stack-v2-derived file IDs and separately retrieved contents | Not selected: Stack-related access/terms and file licenses were not established for this task. No content requested. |

Any resumed corpus task must bound source downloads/cache well below 20 GiB, pin source revisions and files, retain source/license/language/category/provenance for every admitted document, deduplicate content hashes before a document-level split, check disjointness, and report actual language/token mixtures. The desired code/general/technical/structured-context mixture is a target, not a measured result. No arbitrary repository scrape, gating bypass or multi-terabyte download occurred.

Tokenizer decision remains **unknown**. There is no representative real-corpus sample, so tokens/character, tokens/line, fragmentation and pathological cases were not evaluated for Python, C, C++, Rust, Java, Go, JavaScript, TypeScript, HTML/CSS, SQL, Bash, JSON/YAML or English. The old tokenizer is preserved; it has not been declared suitable or unsuitable for real pretraining. If replacement is justified in a resumed task, train a new research_v1/tokenizer.json on real TRAIN documents only, freeze it for every compared architecture, and do not compare loss numerically across tokenizers.

| Requested quantity | State |
|---|---|
| Real raw/usable token count and realized mixture | unknown; corpus not acquired |
| Real document deduplication, train/validation leakage, language statistics | not performed |
| Per-document provenance/license audit | not performed |
| Representative tokenizer evaluation and replacement decision | not performed |
| Existing development shard/tokenizer integrity | verified by Phase 1A integrity check |

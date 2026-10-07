# SparseCoderLab

SparseCoderLab studies sparse transformer architectures and conditional memory for code and repository understanding. The current work compares dense and grouped Top-1 expert models, with and without hashed Ngram memory, on a frozen real-document corpus.

## 75M model comparison

I compared a dense model, a sparse model, and sparse models with 10M or 25M Ngram memory. All four reached 100M tokens under the same data and tokenizer. The 25M-memory model had the lowest validation losses; a quick generation screen favored the 10M model on Python parseability and obvious repetition. That screen did not test whether generated code worked.

I stopped the old training setup before a matched 250M comparison was complete. Dense reached 250M and Sparse reached 180.224M; the two Ngram models remained at 100M. These later checkpoints are exploratory, not a final ranking. The four 100M checkpoints are the common starting point for the next data phase. See the [results and checkpoint record](Documentation/54_prompt3_milestone_results.md).

The repository data pipeline now preserves pinned commits, source/test/docs/build roles and repository-level splits. A small Flask/Requests fixture passed ingestion, token packing, shard verification and tokenizer round-trip checks. The full corpus and training preflight are still in preparation; [pipeline methods and source review](Documentation/55_repository_data_pipeline.md) describe the measured result and remaining work.

## Earlier results

At 100M training tokens, the three seed-42 models reached these held-out negative log-likelihoods (lower is better):

| Model | Mixed | Code | General | Technical |
|---|---:|---:|---:|---:|
| Dense | 3.010259 | 2.517283 | 4.804755 | 3.423348 |
| Sparse | 2.981258 | 2.481819 | 4.775507 | 3.397397 |
| Sparse + Ngram | 2.974749 | 2.476290 | 4.750409 | 3.373976 |

A selected second-seed comparison favored Ngram over Sparse from 50M through 100M tokens. The overall result is mixed: the loss trends are promising on this fixed corpus, but the bounded arithmetic-code probes all failed, expert specialization evidence is weak, and two seeds do not establish broad robustness. These measurements do not demonstrate coding-agent capability or transfer to larger models. See [the research review](Documentation/54_realdata_comparison.md) and [the model card](MODEL_CARD.md).

## Reproduction and evidence

The corpus, tokenizer, evaluation, model, and training details are documented in [Documentation](Documentation/README.md). Small metrics, resolved configurations, and analysis tools are versioned here. Large datasets and training checkpoints remain outside ordinary Git; their identities are recorded by hashes and sizes in the experiment evidence.

The project source is licensed under Apache-2.0. Dataset sources and dependencies have separate terms. See [LICENSE](LICENSE).

Repository: [SparseForge-Lab/SparseCoderLab](https://github.com/SparseForge-Lab/SparseCoderLab).

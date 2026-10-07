# SparseCoderLab

SparseCoderLab studies sparse transformer architectures and conditional memory for code and repository understanding. The current work compares dense and grouped Top-1 expert models, with and without hashed Ngram memory, on a frozen real-document corpus.

## 75M model comparison

I am comparing four versions of the same roughly 75M-parameter model: a dense model, a sparse model, and sparse models with 10M or 25M Ngram memory. They share the same seed-42 data and tokenizer. At 100M tokens, the 25M-memory version had the lowest validation loss. The 10M version produced more parseable code and fewer obvious loops in a quick generation check. Those samples were not run as functional tests.

The dense model is the first to reach 250M tokens, with a mixed validation loss of 2.483956. The other models are still running, so I have not picked a winner. The [full results and checkpoint history](Documentation/54_prompt3_milestone_results.md) include the limits of these measurements.

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

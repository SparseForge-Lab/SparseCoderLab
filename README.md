# SparseCoderLab

SparseCoderLab studies sparse transformer architectures and conditional memory for code and repository understanding. The current work compares dense and grouped Top-1 expert models, with and without hashed Ngram memory, on a frozen real-document corpus.

## Prompt-3 architecture study

The four Prompt-3 candidates use a shared frozen seed-42 data and tokenizer setup. At the 100M-token gate, Sparse + 25M Ngram had the lowest held-out NLL, while Sparse + 10M Ngram had the strongest heuristic parseability/loop screen. Dense is the first candidate to complete the 250M gate (mixed NLL 2.483956); its same-gate comparison is still in progress. No architecture has been selected. See the [Prompt-3 milestone table](Documentation/54_prompt3_milestone_results.md) for the matched results and limitations.

## Prompt-2 comparison (historical)

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

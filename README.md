# SparseCoderLab

SparseCoderLab studies sparse transformer architectures and conditional memory for code and repository understanding. The current work compares dense and grouped Top-1 expert models, with and without hashed Ngram memory, on a frozen real-document corpus.

## 75M model comparison

I compared a dense model, a sparse model, and sparse models with 10M or 25M Ngram memory. All four reached 100M tokens under the same data and tokenizer. The 25M-memory model had the lowest validation losses; a quick generation screen favored the 10M model on Python parseability and obvious repetition. That screen did not test whether generated code worked.

The original training setup ended before a matched 250M comparison was complete. Dense reached 250M and Sparse reached 180.224M. A supplemental 25M-memory continuation then reached 180M, improving mixed/code NLL to 2.5253/2.0461; the 10M-memory model remains at 100M. These later checkpoints are exploratory, not a final ranking. The four 100M checkpoints are the common starting point for the next data phase. See the [results and checkpoint record](Documentation/54_prompt3_milestone_results.md).

The repository data pipeline preserves pinned commits, source/test/docs/build roles and repository-level splits. The earlier [multilingual pilot](Documentation/56_multilingual_repository_pilot.md) packs 7.15M training tokens from 13 projects, with 0.60M validation tokens and 1,533 FIM examples. Ingestion, exact/near-duplicate filtering, FIM reconstruction, token packing, shard verification and exact data-cursor resume checks passed. The tokenizer is unchanged; the expanded corpus and final preflight are documented below.

The expanded 44-repository corpus is frozen at 201.15M training / 30.77M held-out tokens. All four canonical 100M models pass final-data CUDA smoke and exact interrupted resume. The [completed preparation record](Documentation/60_frozen_repository_transition_preflight.md) documents source subsets, measured recipe, coverage, bounded contamination/functional evidence and storage. The [controlled repository comparison](Documentation/61_matched_repository_training.md) is now training all four models sequentially to a matched 250,003,456 total tokens; final evaluation is pending.

## Earlier results

The [completed functional screen](Documentation/57_safe_functional_evaluation.md) now checks the saved Python completions in a bounded WASI runtime. The [implementation review](Documentation/58_implementation_correctness_and_memory.md) documents optimizer/resume fixes, streamed data handling and a 36–42% reduction in measured peak allocated training memory. Short timings do not establish a general throughput improvement.

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

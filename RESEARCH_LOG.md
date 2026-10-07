# Research log

## Small-model corpus comparison

Dense, Sparse and Sparse + Ngram reached 100M training tokens on the same frozen corpus and tokenizer. Their mixed NLL was 3.010259, 2.981258 and 2.974749; code NLL was 2.517283, 2.481819 and 2.476290. A selected Sparse/Ngram replication with seed 1337 also reached 100M and supported a modest late Ngram advantage.

The result remains mixed. All 72 primary arithmetic-code probes failed, and low category mutual information provides weak evidence of semantic expert specialization. The runs establish fixed-corpus loss trends rather than coding-agent capability. See the [comparison review](Documentation/54_realdata_comparison.md) and [model card](MODEL_CARD.md).

A controlled fixed-effective-batch benchmark measured substantially higher throughput with microbatch 8 than with microbatch 2. Grouping changes router auxiliary-loss aggregation, so the speed result did not justify changing optimization semantics during an existing comparison. See [the microbatch measurements](Documentation/50_microbatch_throughput.md).

## 75M-class architecture comparison

Four variants reached the common 100M checkpoint: Dense, Sparse, Sparse + 10M Ngram and Sparse + 25M Ngram. The 25M-memory model had the lowest held-out losses; a bounded generation screen favored the 10M-memory model on Python syntax and obvious repetition. That screen did not measure functional correctness.

Dense later reached 250M and Sparse reached 180.224M before the original comparison ended. These unequal later stops are historical evidence. The four matching 100M snapshots remain the common transition weights. See the [results and checkpoint record](Documentation/54_prompt3_milestone_results.md).

## Additional memory scaling result — 2026-10-07

Sparse + 25M Ngram continued from its full 100M state to 180,002,816 tokens without changing the corpus, tokenizer, batch or schedule. Mixed/code/general/technical NLL reached 2.525322/2.046119/4.237180/2.867805. Turning off the Ngram residual increased mixed/code NLL by 0.104025/0.157506.

The additional 79,994,880 tokens took about 41.6 minutes of recorded loop wall time, with 32,747 tokens/sec and 8.23 GiB peak VRAM. Full validation/ablation finished and the worker exited. CPU checks verified immutable snapshot hashes, finite weights/optimizer moments and resumable scheduler/RNG/cursor state. No functional score or architecture winner is claimed. The [180M record](results/prompt3/sparse25m_180M_results.json) preserves these identities and limits.

The history CSV retained its earlier measurements and received the new row. An omitted empty timeout cell in older rows was restored so the syntax, repetition, throughput and checkpoint columns align with their header. Loss values were unchanged. [Scaling curves](results/prompt3/figures/prototype_scaling.png) show the actual, unmatched later endpoints.

## Repository corpus pilot — 2026-10-07

Thirteen pinned repositories produced 5,895 exported files. Exact/near filtering retained 5,526, which packed into 7,151,137 train tokens and 595,542 validation tokens. The unchanged tokenizer supports existing FIM delimiters; 1,533 of 3,695 eligible files were transformed and every retained record passed reconstruction and tokenizer roundtrips. Packed-stream cursor continuation and physical-shard checks passed on CPU. Sixty-three CPU regression tests passed.

The pilot covers the core programming languages across its two splits, but most languages lack independent validation repositories and C# is currently validation-only. It contains no general-reasoning source. Licensing exceptions, fork families, contamination checks, additional data, the final mixture and model/runtime preflight remain open. A small dependency-context fixture supplies static links and bounded multi-file samples, with repeated-file tokens counted separately. No model has trained on the new pilot. See [the source/pipeline methods](Documentation/55_repository_data_pipeline.md) and [language/token inventory](Documentation/56_multilingual_repository_pilot.md).

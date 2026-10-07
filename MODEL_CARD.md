# Research model card

SparseCoderLab is a research codebase for comparing dense transformers, sparse expert layers, and conditional Ngram memory on code-oriented language modeling. The evaluated models are small research systems, not production coding assistants.

## Evaluated architecture and data

The shared backbone has 9 layers, width 320, 5 query heads, 1 KV head, head dimension 64, and context length 1,024. The dense model uses a 448-unit feed-forward layer. Sparse variants use a 384-unit resident layer and 12 grouped Top-1 experts of width 160 at layers 2, 5, and 8. The Ngram variant adds four 131,072-by-8 tables at layer 1, with orders 2, 2, 3, and 3.

Stored parameter counts are 16,574,400 (Dense), 21,562,560 (Sparse), and 25,767,425 (Sparse + Ngram). Estimated active counts are 16,574,400, 16,493,760, and 16,504,353. Active estimates and the 6N FLOP reference are calculations, not hardware measurements.

Training used a frozen 32,768-token byte BPE tokenizer, a fixed 210,012,790-token training split and 8,393,991-token validation split, seed 42, and 8,192 prediction positions per update. The corpus combines pinned code and general-text sources with repository-level splits, deduplication, provenance, and licensing metadata. Its measured mix was 62.855% code, 17.146% technical, 14.285% general, and 5.715% structured context. See [the corpus report](Documentation/32_research_v1_corpus.md) and [tokenizer report](Documentation/33_research_tokenizer.md).

## Results and limitations

At 100M tokens, mixed/code/general/technical NLL was 3.010259/2.517283/4.804755/3.423348 for Dense, 2.981258/2.481819/4.775507/3.397397 for Sparse, and 2.974749/2.476290/4.750409/3.373976 for Sparse + Ngram. A selected second-seed Sparse/Ngram comparison also favored Ngram at 50M, 70M, and 100M. The evidence remains limited to this corpus and two seeds.

All 72 primary arithmetic-code probe attempts failed. Routing used all experts, but category mutual information was low and does not establish useful semantic specialization. The final assessment is mixed: these models provide language-modeling evidence, not proof of software-engineering or agent behavior. See [the full comparison review](Documentation/54_realdata_comparison.md).

The measured models use a 1,024-token context and received no instruction tuning, tool training, or reinforcement learning. No large-model transfer claim is supported. Checkpoint redistribution rights are unresolved, so checkpoints are not included. Dataset and dependency licenses remain separate from the Apache-2.0 source license.

## 75M-class comparison

The later comparison uses 16 layers, width 512, eight query heads, two KV heads and the same frozen vocabulary/context. Stored/estimated active counts are 54,018,560/54,018,560 for Dense, 76,063,232/54,436,352 for Sparse, 86,565,889/54,453,281 for Sparse + 10M Ngram, and 101,245,953/54,453,281 for Sparse + 25M Ngram. The memory counts include the conditional tables. These active counts are much larger than the eventual few-million-parameter miniature aspiration.

All four have matching 100M checkpoints. A historical 25M-memory continuation reached 180,002,816 tokens with mixed/code/general/technical NLL of 2.525322/2.046119/4.237180/2.867805. Ngram residual-off increased mixed/code NLL by 0.104025/0.157506. This is a single-seed prototype-corpus scaling result with no functional coding score at 180M. Its measured throughput was 32,747 tokens/sec and peak allocation 8.23 GiB on the RTX 5070. Checkpoint hashes and resumability evidence are in the [180M record](results/prompt3/sparse25m_180M_results.json).

The canonical transition weights remain the four 100M snapshots; no winner is selected. A separate [multilingual repository pilot](Documentation/56_multilingual_repository_pilot.md) contains 7.15M train tokens, 0.60M validation tokens and 1,533 FIM examples from pinned repositories. No model has trained on that pilot, and its licensing/fork/contamination review, final mixture and training preflight remain incomplete.

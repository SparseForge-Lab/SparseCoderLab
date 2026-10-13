# Research model card

SparseCoderLab is a research codebase for comparing dense transformers, sparse expert layers, and conditional Ngram memory on code-oriented language modeling. The evaluated models are small research systems, not production coding assistants.

## Earlier small-model architecture and data

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

The four 100M snapshots supplied the canonical transition weights; no winner was selected at that stage. A separate [multilingual repository pilot](Documentation/56_multilingual_repository_pilot.md) contains 7.15M train tokens, 0.60M validation tokens and 1,533 FIM examples from pinned repositories. This was a preprocessing pilot rather than a model-training run; its report records the review and preflight gaps present at that stage. The subsequent expanded corpus and completed preflight are documented in [the frozen transition record](Documentation/60_frozen_repository_transition_preflight.md).

## Repository-data continuation

All four 75M-class variants completed a matched comparison at 250,003,456 total tokens on the expanded frozen repository corpus. The [matched report](Documentation/61_matched_repository_training.md) records loss, functional scores and checkpoint checks. The [truncation follow-up](Documentation/63_prompt5_functional_cap_followup.md) is a separate diagnostic evaluation; its partial 512-token results are not a matched four-model comparison.

Dense, Sparse + 10M Ngram and Sparse + 25M Ngram have completed the matched **499,998,720-token** continuation and evaluation. Plain Sparse was intentionally excluded from the continuation. The [500M report](Documentation/64_matched_500m_results.md) keeps matched executable scores (0/175, 4/175, 4/175), strict target-function scores (0/175, 3/175, 9/175) and whole-output scores (0/175, 5/175, 3/175) separate. +25M has the lowest mixed NLL, 2.631813498392804; no final architecture winner is established. Both Ngram variants are retained for a planned Dataset V2 comparison at 625,000,448 and 750,002,176 tokens. Dataset V2 construction is in progress through the [foundation and pilot pipeline](Documentation/dataset_v2_and_forgebuilder.md). That student training has not started.

These single-seed 25-prompt/seven-temperature results do not establish general coding-agent capability. Excess continuation remains common, especially for +25M. Target extraction and whole-output acceptance use different constraints and are not nested metrics. See the linked report for scoring settings, inferred stopping limitations, residual-off evidence and exact checkpoint identities.

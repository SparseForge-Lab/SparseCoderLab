# Research status

## Current repository-data comparison

The matched four-model comparison at 250,003,456 tokens and the selected three-arm continuation are complete. Dense, Sparse + 10M Ngram and Sparse + 25M Ngram have completed the matched **499,998,720-token** continuation and evaluation. Plain Sparse was intentionally excluded from the continuation. The [500M report](Documentation/64_matched_500m_results.md) keeps matched executable scores (0/175, 4/175, 4/175), strict target-function scores (0/175, 3/175, 9/175) and whole-output scores (0/175, 5/175, 3/175) separate. +25M has the lowest mixed NLL, 2.631813498392804; no final architecture winner is established. Both Ngram variants are retained for a planned Dataset V2 comparison at 625,000,448 and 750,002,176 tokens. Dataset V2 construction is in progress through the [foundation and pilot pipeline](Documentation/dataset_v2_and_forgebuilder.md). That student training has not started.

## Earlier 100M real-data comparison

The real-data architecture comparison is complete through 100M tokens for Dense, Sparse, and Sparse + Ngram at seed 42. A selected Sparse/Ngram second-seed comparison is complete at 20M, 50M, 70M, and 100M. Checkpoint integrity, frozen-input identity, evaluations, routing coverage, ablations, and release records passed their final checks.

The result is **MIXED**. Sparse improves held-out language-modeling loss over Dense by 100M tokens, and Ngram provides a small late-stage gain over Sparse in both tested seeds. All 72 primary arithmetic-code probes failed, evidence for semantic expert specialization is weak, and the available seeds do not support broad robustness claims. These results do not demonstrate coding-agent capability or transfer to larger models.

The highest-value follow-up is stronger executable repository evaluation and additional independent-seed comparisons. See [the research review](Documentation/54_realdata_comparison.md), [the model card](MODEL_CARD.md), and the machine-readable evidence under `results/`.

# Research status

The real-data architecture comparison is complete through 100M tokens for Dense, Sparse, and Sparse + Ngram at seed 42. A selected Sparse/Ngram second-seed comparison is complete at 20M, 50M, 70M, and 100M. Checkpoint integrity, frozen-input identity, evaluations, routing coverage, ablations, and release records passed their final checks.

The result is **MIXED**. Sparse improves held-out language-modeling loss over Dense by 100M tokens, and Ngram provides a small late-stage gain over Sparse in both tested seeds. All 72 primary arithmetic-code probes failed, evidence for semantic expert specialization is weak, and the available seeds do not support broad robustness claims. These results do not demonstrate coding-agent capability or transfer to larger models.

The highest-value follow-up is stronger executable repository evaluation and additional independent-seed comparisons. See [the research review](Documentation/54_realdata_comparison.md), [the model card](MODEL_CARD.md), and the machine-readable evidence under `results/`.

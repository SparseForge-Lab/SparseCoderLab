# Research status

## Current repository-data comparison

The matched four-model comparison at 250,003,456 total training tokens is complete. The continuation targets 499,998,720 tokens. Sparse + 10M Ngram reached a verified 286,720,000-token checkpoint before stopping; Sparse + 25M Ngram has not continued yet. The Dense midpoint checkpoint reached 374,996,992 tokens; the final matched evaluation remains pending. See the [250M comparison](Documentation/61_matched_repository_training.md), [truncation follow-up](Documentation/63_prompt5_functional_cap_followup.md) and [375M interim report](Documentation/54_prompt5_500m_interim.md). These are small-model results and do not establish general coding ability.

## Earlier 100M real-data comparison

The real-data architecture comparison is complete through 100M tokens for Dense, Sparse, and Sparse + Ngram at seed 42. A selected Sparse/Ngram second-seed comparison is complete at 20M, 50M, 70M, and 100M. Checkpoint integrity, frozen-input identity, evaluations, routing coverage, ablations, and release records passed their final checks.

The result is **MIXED**. Sparse improves held-out language-modeling loss over Dense by 100M tokens, and Ngram provides a small late-stage gain over Sparse in both tested seeds. All 72 primary arithmetic-code probes failed, evidence for semantic expert specialization is weak, and the available seeds do not support broad robustness claims. These results do not demonstrate coding-agent capability or transfer to larger models.

The highest-value follow-up is stronger executable repository evaluation and additional independent-seed comparisons. See [the research review](Documentation/54_realdata_comparison.md), [the model card](MODEL_CARD.md), and the machine-readable evidence under `results/`.

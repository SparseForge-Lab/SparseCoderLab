# Repository-data continuation checkpoints

The Dense 75M reference continued from its matched 250,003,456-token full-state checkpoint to 374,996,992 tokens. The resulting checkpoint passed full-state verification, including optimizer and RNG state. Its SHA-256 is `46603a0f274e4601f25488c53e65fe8137954e3ca77068a3e959d5a1b1c3381c`.

The generation benchmark contains 25 fixed prompts at seven temperatures, with all 175 outputs retained. The readable report and raw rows are in `results/prompt5_500M_v1/milestone_benchmarks/`. Repetition and Python parseability checks are quick diagnostics; generated code was not executed in this benchmark, so these checks are not functional results.

Sparse + 10M Ngram continued from its matched 250,003,456-token checkpoint to 286,720,000 tokens (step 35,000). The verified full-state checkpoint includes optimizer and RNG state; its SHA-256 is `fc63cfaee63e5c098541041f138d233e64f0b28091770d66db24eae85fef2583`. The run stopped after saving this checkpoint.

Sparse + 25M Ngram has not continued yet. Dense and Sparse + 10M Ngram have not both reached the next matched milestone, so a fair comparison remains pending. The target is 499,998,720 tokens followed by the final matched evaluation. This report records interim results.

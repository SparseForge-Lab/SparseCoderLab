# Historical repository-data continuation checkpoints

Status: historical interim evidence. All three selected arms later completed 499,998,720 tokens and matched evaluation; see the [final 500M report](64_matched_500m_results.md). The chronology below describes earlier states, not the current run status.

The Dense 75M reference continued from its matched 250,003,456-token full-state checkpoint to 374,996,992 tokens. The resulting checkpoint passed full-state verification, including optimizer and RNG state. Its SHA-256 is `46603a0f274e4601f25488c53e65fe8137954e3ca77068a3e959d5a1b1c3381c`.

The generation benchmark contains 25 fixed prompts at seven temperatures, with all 175 outputs retained. The readable report and raw rows are in `results/prompt5_500M_v1/milestone_benchmarks/`. Repetition and Python parseability checks are quick diagnostics; generated code was not executed in this benchmark, so these checks are not functional results.

Sparse + 10M Ngram continued from its matched 250,003,456-token checkpoint to 286,720,000 tokens (step 35,000). The verified full-state checkpoint includes optimizer and RNG state; its SHA-256 is `fc63cfaee63e5c098541041f138d233e64f0b28091770d66db24eae85fef2583`. At that point, the run stopped after saving this checkpoint; it subsequently resumed and reached the final gate.

At this historical checkpoint, Sparse + 25M Ngram was still queued and the common final comparison was pending. Those conditions are superseded by the verified completed 500M comparison linked above. The checkpoint hashes and diagnostic observations here are retained as historical evidence.

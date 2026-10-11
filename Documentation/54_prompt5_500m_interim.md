# I saved the Prompt-5 continuation checkpoint

I continued the Dense 75M reference from its matched 250,003,456-token full-state checkpoint to 374,996,992 tokens. I verified the resulting full checkpoint, including its optimizer and RNG state. Its SHA-256 is `46603a0f274e4601f25488c53e65fe8137954e3ca77068a3e959d5a1b1c3381c`.

I ran the same 25 fixed prompts at seven temperatures and saved all 175 outputs. The readable report and raw rows are in `results/prompt5_500M_v1/milestone_benchmarks/`. The report includes repetition and Python parseability checks, which are only quick diagnostics; I did not run the generated code or count those checks as functional results.

I continued Sparse + 10M Ngram from its matched 250,003,456-token checkpoint and saved a verified full-state checkpoint at 286,720,000 tokens (step 35,000). It includes optimizer and RNG state; its SHA-256 is `fc63cfaee63e5c098541041f138d233e64f0b28091770d66db24eae85fef2583`. I stopped the run after saving it. Sparse + 25M has not continued, and the Dense and Sparse + 10M models have not both reached the next matched milestone, so I cannot compare them fairly yet. The final 499,998,720-token continuation and matched evaluation are still to come. This is an interim result.

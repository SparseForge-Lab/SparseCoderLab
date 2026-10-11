# I reached the 375M checkpoint

I continued the Dense 75M reference from its matched 250,003,456-token full-state checkpoint to 374,996,992 tokens. I verified the resulting full checkpoint, including its optimizer and RNG state. Its SHA-256 is `46603a0f274e4601f25488c53e65fe8137954e3ca77068a3e959d5a1b1c3381c`.

I ran the same 25 fixed prompts at seven temperatures and saved all 175 outputs. The readable report and raw rows are in `results/prompt5_500M_v1/milestone_benchmarks/`. The report includes repetition and Python parseability checks, which are only quick diagnostics; I did not run the generated code or count those checks as functional results.

The Sparse + 10M and Sparse + 25M continuations have not both reached this checkpoint, so I cannot compare the three models fairly yet. I am continuing all three toward 499,998,720 tokens, then I will run the matched final evaluation. This is an interim result.

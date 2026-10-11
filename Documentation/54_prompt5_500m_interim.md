# I got the first Prompt-5 run to 375M

I got the Dense 75M continuation to 374,996,992 tokens and verified that its saved checkpoint still contains the full training state. Its full checkpoint SHA-256 is `46603a0f274e4601f25488c53e65fe8137954e3ca77068a3e959d5a1b1c3381c`.

I ran the same 25 prompts at seven temperatures and saved all 175 outputs, along with a readable summary, under `results/prompt5_500M_v1/milestone_benchmarks/`. The report includes quick generation heuristics such as repetition and Python parseability; those are only diagnostics, not functional test results. I have not executed generated code.

The other two runs still need to reach this checkpoint before I can compare the three models fairly. I am continuing those runs toward 500M and will run the matched final evaluation after they finish. This is an interim checkpoint, not the final Prompt-5 result.

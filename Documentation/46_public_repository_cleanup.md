# Public repository cleanup during active research

Public research content includes source, configs, tests, architecture/protocol/results/limitations documentation, corpus/tokenizer provenance, small statistics, immutable checkpoint references, model-card evidence and reproducibility scripts. The cleanup also publishes the completed all-three20M result, including memory ablation and descriptive routing evidence, without treating it as a final architecture decision.

Removed from the current public Git tree while retaining canonical local files:

- GOALS/Prompt-1.md
- GOALS/Prompt-2.md
- GOALS/README.md
- web/local_dense_chat.html
- results/history/Prompt-2/sparse_uncheckpointed_attempt/metrics.jsonl

Raw Documentation/00_* request files were already untracked and remain private. Entire GOALS/web folders stay ignored by the user's explicit earlier instruction; useful research documentation produced from those requests remains public. Normal cleanup does not rewrite earlier Git history.

Removed the broad experiments/*/ exclusion. Small experiment config.json, provenance.json and summary*.json records are now public. Precise rules still exclude model formats, checkpoints, raw metrics, process-output logs, caches, raw/downloaded data and tokenized shards. Live schedule_state.json and the raw user-resume/process receipt stay local. operating_conditions.json publishes the research-relevant concurrent-use scope without task text or process IDs.

The permanent policy is recorded in Documentation42: classify files before staging; preserve useful reproduction/research evidence; ignore private/local/large artifacts before committing; review and scan the exact index; keep local copies when untracking. Offline archives retain private goals, requests, web assets and small research logs independently from the public Git exclusions.

The live Ngram20M→50M training worker is left running throughout cleanup. No stop, pause, restart, milestone replay, corpus/tokenizer/config change or new research phase is introduced. Roblox remains open. Ngram20M checkpoint and evaluation are complete; all-three50M review and remaining70M/100M schedule stay authoritative. Actual push/continuity verification is reported separately after publication.

# SparseCoderLab

Public repository: [SparseForge-Lab/SparseCoderLab](https://github.com/SparseForge-Lab/SparseCoderLab). Project code is open source under [Apache-2.0](LICENSE). Dataset records, dependencies and external model artifacts retain their respective licenses; this license does not relicense the training corpus.

A runnable RTX 5070 architecture lab for cheaply falsifying hypotheses before scaling a sparse coding/agent model. The micro models are research instruments, not coding agents.

Prompt-2.5 continues the same Prompt-2 comparison after shutdown. DenseCompute and grouped SparseV3 now have verified20M/50M checkpoints. Training is stopped at the user's request after GitHub preparation; Ngram and all70M/100M stages remain pending. Complete-update endpoints20,004,864 /50,003,968 /70,000,640 /100,007,936 tokens. Final GOOD/MIXED/BAD classification remains pending.

All variants share9 layers,width320,context1024 and frozen32768 real tokenizer. Dense16.57M stored is the active-compute control; sparse21.56M adds12x160 capacity at layers2/5/8; Ngram25.77M adds current4.19M table values. Inputs, seed42, optimizer and cosine schedule match. Real TRAIN corpus210M unique tokens, pinned bounded sources, repository/hostname splits and audited provenance. This micro stage does not prove60–75B scaling.

Latest50M: Dense mixed NLL3.288592/code2.778863; Sparse mixed3.276931/code2.756853. Sparse-Dense changed from+0.161791 mixed at20M to−0.011661 at50M. This small single-seed advantage needs the remaining curves and replication decision; it is not a final architecture verdict. Code diagnostic differs from corpus language prevalence; dense cumulative wall is a documented lower bound. Read `FINAL_STATUS.md`, `MODEL_CARD.md`, `Documentation/README.md`, `results/research_history.csv`, `results/model_card_evidence.json` and `results/releases/` for verified scopes and checkpoint references.

```powershell
Set-Location D:\SparseCoderLab
.\setup.ps1
.\.venv\Scripts\python.exe -m tools.prepare_data
.\.venv\Scripts\python.exe -m tools.count_params
.\.venv\Scripts\python.exe -m pytest -q
```

Data/tokenizer already prepared locally should not be overwritten: preparation rejects an existing frozen shard manifest. All configuration comes from YAML; variants inherit the same base. CUDA is mandatory for training. Native SDPA/eager code is the reference. WSL is optional and currently unavailable on this machine. Existing installations and Strata are not modified.

Historical engine cap255 minutes; research cumulative stages cap300 minutes. Checkpoint/resume retains model/optimizer/scheduler/RNG/cursor; atomic writes and immutable milestones are used. Original synthetic fixtures remain engineering regressions; research_v1 is the frozen real-data quality lane. Prompt-2 long training is explicitly authorized; no subsequent prompt starts automatically.

For frozen research continuation, verify `python -m tools.research_resume_state`, then the GPU resume check, Phase0 and research gate. Use `python -m tools.research_train --model sparse --endpoint 20004864`; then sparse50M, memory20M/50M, matched review, all70M and all100M. Completed milestones are verified/skipped, not retrained. Documentation34 has exact protocol and exceptions; Documentation38 has permanent publication-evidence rules. The optional local browser playground starts with `python -m tools.local_dense_chat` at http://127.0.0.1:8765 and uses CPU inference.

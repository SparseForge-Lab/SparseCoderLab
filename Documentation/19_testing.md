# Tests and evidence

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=results/tests.xml
.\.venv\Scripts\python.exe -m tools.overfit --config configs/dense_compute.yaml
.\.venv\Scripts\python.exe -m tools.profile --config configs/dense_compute.yaml --seconds 300
.\.venv\Scripts\python.exe -m tools.profile --config configs/sparse_v3.yaml --seconds 30
.\.venv\Scripts\python.exe -m tools.profile --config configs/sparse_memory.yaml --seconds 30
.\.venv\Scripts\python.exe -m tools.phase0
```

Tests cover tokenizer/whitespace/Unicode roundtrip, document leakage, scalar-reference hash and GPU parity, causal lookup gradients, MoE shape/Top1/Top2/router gradients/balance collapse, dense/sparse causal forward/backward, MTP shifts/shared vs independent steps/generation, exact checkpoint scheduler/RNG continuation, dataloader cursor, exact parameter arithmetic, CPU rejection and RTX BF16, archive integrity, trace serialization, graph packing, hand-computed tier hits/transfers, RouteAhead shapes/metrics, large projections and sandbox path/loop behavior.

No test is reported as passed until executed. JUnit reports and captured output are the evidence. GPU tests require CUDA and fail if unavailable. Overfit checks the actual DenseCompute dimensions over a tiny repeating sequence; it proves learning works, not meaningful generalization. Long runs are gated against current source; optional experimental failures must remain visible and block promotion where relevant.

Prompt-1 adds eight grouped-MoE regressions, bringing the executed suite to40. `results/moe_regression_tests.*` and `results/moe_phase0_output.txt` preserve the passing runs; previous tests/gate are retained under history/Prompt-1. The separate source-bound `python -m tools.moe_parity --deterministic` gate covers five-update full-model FP32/BF16 and old trained checkpoint/AdamW parity. See Documentation30 for exact tolerances, baseline repeatability controls and the rejected padded backward. The unchanged historical five-minute dense/overfit profiles remain engineering evidence in the Phase0 gate, not newly repeated quality experiments.

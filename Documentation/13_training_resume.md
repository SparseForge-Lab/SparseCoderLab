# Training, timing and exact restart

```powershell
.\.venv\Scripts\python.exe -m tools.train --config configs/dense_compute.yaml --run-dir experiments/dense_first --target-tokens 5000000 --max-wall-minutes 20
.\.venv\Scripts\python.exe -m tools.train --config configs/dense_compute.yaml --run-dir experiments/dense_first --resume experiments/dense_first/checkpoints/last.pt --target-tokens 5000000 --max-wall-minutes 20
```

Ctrl+C requests a clean stop. The current gradient-accumulated optimizer step finishes, then the checkpoint is atomically written using a flushed temporary file and replace. The wall timer reserves time for checkpointing; validation runs only if there is sufficient remaining time. Default cap is 255 minutes, leaving the rest of a five-hour GPU allocation for validation/storage.

Checkpoint state: model, AdamW optimizer, LR scheduler, Python/NumPy/torch/CUDA RNG, packed-stream cursor, token count, step count, cumulative wall/training time, seed, config hash, data manifest hash, tokenizer hash and git revision. BF16 uses no GradScaler; its checkpoint field is explicitly null. Resume rejects changed config/data/tokenizer. `target_tokens` is a cumulative limit, not an additional-token count.

Data cursor counts consumed packed rows and reconstructs each epoch's seeded permutation. Gradient accumulation is completed before saving, so partial gradients do not need serialization. Regression tests compare the next random batch, exact CPU tensors/scheduler and actual GPU training continuation. Deterministic evaluation starts from the same validation permutation.

Each run writes config, provenance, JSONL metrics, summary and leaderboard row. Metrics include loss, gradient norm, tokens/sec, utilization when available, VRAM peak, loads/entropy and MTP horizon metrics. GPU utilization is sampled, not a full utilization integral. Checkpoint budgeting includes a temporary replacement allowance; only `last.pt` is retained, avoiding cherry-picked best checkpoints.

Compile is blocked until a separate correctness lane exists. BF16 and TF32 are configured. Activation checkpointing is selectable; eager correctness comes first. The batch finder benchmarks candidate microbatches with headroom, records OOM and recommends a batch; it does not silently change fairness configs.

# Phase 1A results — stop/review

All three fresh seed-42 runs completed 611 full optimizer steps, exactly 5,005,312 tokens each. The nominal 5M endpoint was rounded upward before training to preserve identical full updates. Context 1024, microbatch 2, accumulation 4, token order/data seed 42, frozen original 32,768 tokenizer, BF16 native causal SDPA, AdamW, original warmup/LR schedule and evaluation points 200/400/600/611 match. Top1 routing; no MTP, RouteAhead, locality loss, alternative attention or compaction training. No GPU jobs overlapped.

Validation uses 16 fixed mixed batches (32,768 predictions) and the first 16 held-out documents in each code/general subset, capped at context+1. Those subset means cover different tokens from the mixed estimate. They cannot be added to reconstruct the mixed loss. The fixtures share repetitive templates across disjoint document hashes. This is an engineering comparison only.

Wall time starts after model/optimizer/data initialization and includes logging, evaluations and checkpoint writes. CUDA verification/setup is excluded. Step throughput excludes non-step overhead, except sampled memory diagnostics are deliberately inside timed steps. Peak VRAM is PyTorch allocated memory, not total driver-visible usage. Results are one sequential observation per model, without timing replications.

| Model | Validation NLL | Code NLL | General NLL | Bits/token | Wall s | Wall tok/s | Step tok/s | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DenseCompute | 4.349110 | 3.984163 | 5.071468 | 6.274439 | 184.80 | 27085 | 30324 | 1.476 |
| SparseV3 | 4.345538 | 3.980094 | 5.073375 | 6.269287 | 492.90 | 10155 | 10874 | 1.566 |
| SparseV3Ngram | 4.349094 | 3.982275 | 5.079932 | 6.274416 | 527.25 | 9493 | 10174 | 1.636 |

| Quantity | Evidence label | Scope |
|---|---|---|
| Losses, wall time, throughput, peak allocated memory, gradient/gate/router observations | measured | Actual completed CUDA runs; diagnostic gradients/routes/gates sampled every 10 steps |
| Stored parameters | calculated | Exact sum of instantiated parameter sizes |
| Active parameter estimates and 6N training FLOPs | calculated | Reference arithmetic; excludes quadratic attention, dispatch, lookup arithmetic and hardware efficiency |
| Bits/token, CV, means, collision counts and speed ratios | calculated from measurements | No independent hardware FLOP measurement |
| Real coding quality, across-seed reproducibility, useful specialization/locality, large-model behavior | unknown | Research corpus, Phase 1B and larger trace phase were not run |

SparseV3 minus dense mixed NLL is -0.003571; Ngram minus dense is -0.000016. Sparse's gap changes sign at the scheduled checkpoints. These tiny single-seed differences do not establish an architecture winner or growing capacity advantage.

At equal tokens, SparseV3 takes **2.67x wall time** and Ngram **2.85x**. On training steps alone, the ratios are **2.79x** and **2.98x**. These measured slowdowns are a negative result despite the approximately matched active estimates. DenseCompute is the fairest active-compute reference; DenseSize would be an optional stored-capacity reference, and was not run here.

All microbatch losses and complete-model gradient norms passed finite checks. Sampled pre-clipping global gradient ranges are dense 0.404–2.984, sparse 0.359–4.782, memory 0.447–7.535. Spikes were clipped at 1.0. Early hard-routing concentration recovered. Mean sampled router entropy/CV are sparse 2.311/0.526 and memory 2.323/0.538. The entropy is soft router-probability entropy in nats, not entropy of hard assignments. The late 20-sample maximum expert shares stay below 18.5%, rather than showing persistent collapse. Balanced routing does not prove specialization.

The explicit user stop/review condition applies: dramatic sparse slowdown without an established quality benefit. **No corpus acquisition, Phase 1B C1/C2, second seed, memory-capacity training or new routing-trace phase follows automatically.** The memory-specific measurements are in 26_phase1a_ngram.md; missing phases are explicitly reported in 24/25/27.

Evidence: results/phase1a_comparison.csv, phase1a_curves.csv, phase1a_diagnostics.json, phase1a_ngram_ablation.json, phase1a_integrity.json and phase1_review_decision.json. Complete raw metrics/configs/provenance/summaries and checkpoints are under experiments/phase1a_dense, phase1a_sparse and phase1a_memory. Reproduce postprocessing with `.venv\Scripts\python.exe -m tools.phase1_collect`, `-m tools.phase1_integrity` and `-m tools.phase1_ablation` from D:\SparseCoderLab. The ablation command uses the existing trained local checkpoint and performs no training.

The original Phase 0 gate passed 30 tests before edits. Instrumentation/ablation tests then brought the passing suite to 32. Original status, measurements, gate/test evidence and leaderboard are preserved in results/history/pre_phase1. Training used source hash 4013f064397ea25b7fbb5aa0e4ccf4a1baa58ac00dee9f933af1759b136ca72b; byte-exact source is archived in results/history/phase1a_source.zip with results/phase1a_source_manifest.json. Postprocessing tools were added after the runs, so the current-source gate may have a different hash. Original YAMLs, tokenizer and shards were not changed.

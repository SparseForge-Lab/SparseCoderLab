# Large-model projection

`tools/project_large_model.py` takes dimensions, layers/capacity layers, experts, routed/resident FFNs, Top-k, memory capacity, precision, GPU/RAM and SSD/cache assumptions. It reports stored/active parameters, BF16/FP8/INT8/INT4 ideal weight bytes, expert/page bytes, worst/cached bytes per token, SSD-only ceiling, host-RAM requirement and VRAM coverage after resident/activation reserves.

`configs/large_candidate.yaml` is one example: d_model 2560, 48 blocks, 16 capacity layers, 256 experts, routed FFN 1536. Additional unspecified dimensions are explicit assumptions: resident FFN 3072, 40 Q/8 KV heads, 64 head dimension, 32,768 vocabulary. Conditional memory sweeps 8B/12B/16B/20B/25B; Top1 and Top2 both run. These settings are not a selected winner.

```powershell
.\.venv\Scripts\python.exe -m tools.project_large_model --precision int4 --ssd-mib-s 462 --cache-hit-rate 0.5
```

That hit rate is illustrative unless measured for the projected topology. Quantization metadata, dequantization cost, allocator overhead, KV cache and storage latency need separate estimates. Whole-page fetch can overfetch. Mini-model SSD throughput does not establish 60B decode throughput; arithmetic can instead falsify impossible bandwidth/RAM targets.

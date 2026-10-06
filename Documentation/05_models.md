# Models and exact parameter counts

All three backbones share 320 hidden dimensions, 9 layers, 5 query heads, 1 KV head, 64 head dimension, RMSNorm, RoPE and native causal SDPA. KV heads are explicitly repeated for GQA. Bias-free SwiGLU has three matrices, so its parameter count is `3 * d_model * ffn`. The embedding/output weight is tied and counted once.

| Configuration | Stored parameters | Active estimate/token |
|---|---:|---:|
| DenseCompute, FFN 448 | 16,574,400 | 16,574,400 |
| DenseSize, FFN 1536 | 25,974,720 | 25,974,720 |
| SparseV3, resident FFN 384, Top1 | 21,562,560 | 16,493,760 |
| SparseV3 + N-gram | 25,767,425 | 16,504,353 |
| SparseV3, Top2 | 21,562,560 | 16,954,560 |
| SparseV3 + shared MTP | 21,921,280 | 16,852,480 |

Sparse capacity layers are zero-based 2, 5 and 8. Each has 12 experts with intermediate width 160, added to the resident FFN. Routed tables total 5,529,600 parameters; routers add 11,520. The ~25.7M size includes the optional N-gram table. SparseV3 without it is ~21.56M; this is an arithmetic finding, not an implementation adjustment to force the target.

`python -m tools.count_params` constructs the actual modules and counts unique trainable tensors, separating experts, routers, resident weights, memory and MTP. Active estimates subtract inactive experts and unused memory rows but include the dense tied output head. This measures approximate parameter access, not FLOPs or throughput. Shared MTP weights are counted once but execute three times.

DenseCompute versus SparseV3 tests similar active compute with different stored capacity. DenseSize versus SparseV3+memory tests similar stored size with different active compute. Same tokenizer, data order, seed, optimizer, context and attention are mandatory; measured dispatch overhead can defeat nominal compute savings.

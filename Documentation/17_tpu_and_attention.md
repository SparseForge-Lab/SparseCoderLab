# TPU compatibility and future attention lane

The PyTorch prototype isolates backbone layers, conditional memory, routers, draft blocks and training/data state. This makes mathematical behavior transferable; it is not a JAX port. Eager token-by-token expert dispatch is a reference and must be replaced with grouped MoE primitives for TPU.

Future MaxText/JAX concerns: static shapes and padding for dynamic routing; grouped GEMM scheduling; expert-parallel All-to-All; capacity/load balance; N-gram lookup placement; host prefetch latency; checkpoint conversion; identical tokenizer/data/target shift and seed semantics. Custom sparse attention or state mixers need correctness and isolated quality tests before distributed optimization. Aim for MaxText-compatible grouped MoE and measure actual TPU MFU. RTX utilization/MFU does not predict TPU MFU.

Attention remains identical native SDPA for the initial Dense/Sparse comparison. A separate lane can evaluate state-space/linear mixers, sparse/block attention and local+retrieval attention. Promote candidates only after matched-quality/latency tests at 1k/2k/4k and targeted 8k. No third-party kernel or 256k training is required in V1.

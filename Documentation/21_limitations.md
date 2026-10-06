# What remains experimental

Implemented reference mechanisms do not imply successful hypotheses. No real-code pretraining, two-seed architecture ranking, useful specialization/locality, MTP throughput advantage, RouteAhead latency improvement or neural compaction retention has been established. No long run is launched automatically.

The synthetic corpus is template-rich and tiny. Its tokenizer is a development tokenizer, frozen for current comparisons; a production corpus may warrant a new research campaign with one newly frozen tokenizer. The eager expert loop can be slower than dense despite similar active-parameter arithmetic. The offload model is a simulator, not asynchronous physical SSD/RAM/VRAM serving. Large projections depend on unverified cache/topology assumptions.

Compaction dictionary retention and exact archive recovery are infrastructure results. They do not establish a model's ability to retrieve/select/use facts. Teacher target generation/SFT is disabled. Design B MTP, page-aware routing, third-party attention parity, JAX/TPU training, optimized KV-cached speculation and production secure tool sandboxes are future lanes.

The tracked goal concerns a runnable, documented, measured research laboratory, not proof that any architecture wins. FINAL_STATUS reports engineering readiness and remaining experiments explicitly.

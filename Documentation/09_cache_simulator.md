# VRAM / RAM / SSD simulation

`src/runtime/cache.py` accepts a routing trace, expert-to-page mapping, quantized expert bytes, fixed VRAM/RAM byte capacities and bandwidth/latency assumptions. Demand misses load from RAM when present and from SSD otherwise. Cache policies include LRU, LFU and pinned frequent pages. Oversized pages cannot fit a tier. Outputs include hit rates, SSD bytes/token, RAM-to-VRAM bytes/token, page reads/token, I/O stall/token, I/O-only throughput ceiling, churn and prefetch useful/wasted bytes.

This is a serial-transfer reference model, not a physical offload runtime. It excludes kernel concurrency, quantization/dequantization, PCIe overlap details, storage queue effects and model compute. Prefetch has a shared finite time window; wasted reads consume bandwidth. Predictor accuracy alone is insufficient: demand latency plus wasted traffic must improve relative to the same cache baseline.

`tools.routing_lab --ssd-mib-s <measured bandwidth> --ssd-latency-ms <independent residual latency>` records explicit storage assumptions. The environment probe's 4 MiB block latency includes transfer; do not add it again. Default config values are assumptions, not benchmark results. On this machine the source device is SATA, although the abstract model supports NVMe bandwidth inputs.

Large-model use must change expert size, capacity-layer count, expert population and cache coverage. Micro traces represent a different expert population. Any remapped/repeated scaling trace is a sensitivity scenario, never real evidence of future 60–75B routing or throughput.

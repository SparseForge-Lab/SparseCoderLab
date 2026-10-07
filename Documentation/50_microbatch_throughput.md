# Microbatch throughput measurements

A fixed-effective-batch benchmark compared Sparse and Sparse + Ngram at four real microbatch/accumulation settings. Each update covered 8,192 prediction positions. The test used fresh speed-only models, three warmup updates, and two reversed-order 20-update measurements per setting. It recorded wall throughput, step time, allocated/reserved VRAM, host batch preparation time, optimizer time, and process CPU use.

| Architecture | 1 × 8 | 2 × 4 | 4 × 2 | 8 × 1 | 8 × 1 / 2 × 4 |
|---|---:|---:|---:|---:|---:|
| Sparse | 13,727 | 25,799 | 45,034 | 73,359 | 2.84× |
| Sparse + Ngram | 13,110 | 24,345 | 43,588 | 72,563 | 2.98× |

Rates are tokens per second, averaged over two short runs. At 8 × 1, peak reserved memory was 6.12 GiB for Sparse and 6.07 GiB for Sparse + Ngram on the 12 GiB RTX 5070. The largest tested microbatch was fastest in both repetitions.

These are short throughput measurements, not evidence of equal model quality, long-run stability, or exact checkpoint continuation. MoE balance loss depends on the real microbatch, so changing microbatch changes the optimization behavior even when the effective token batch is held fixed. The next training lifecycle should measure quality and stability under the selected setting before relying on the full short-run speed ratio.

Raw measurements and benchmark configuration are in `results/research_batch_tuning/`. The benchmark was run after the comparison training had stopped and wrote no quality checkpoints or dataset files.

# Bounded runtime measurement protocol

All GPU jobs run sequentially on the RTX5070; none overlap with another project GPU test/profiler/benchmark. Native Windows Python 3.12.10, isolated torch 2.14.1+cu130, BF16/native SDPA and TF32 enabled match the existing speed policy. Correctness separately uses deterministic algorithms and TF32 disabled. No dependency installation, corpus acquisition, checkpoint write or long quality training occurs.

The layer matrix contains 96 rows: contexts 256/1024/2048, microbatch 1/2/4/8, balanced/moderate/strong/natural distributions, and reference/grouped. Batch8 is the largest configured useful candidate tested, not an assertion of the absolute physical maximum. It completes without OOM. Each function has three warm calls and ten timed calls; report medians of synchronized host elapsed time and CUDA-event elapsed time. Forward excludes autograd; forward+backward includes gradient reset and a small squared-output plus balance objective. It is a layer measurement rather than language-model training throughput.

Natural inputs are the capacity-layer-2 normalized hidden states captured from the actual trained SparseV3 checkpoint on fixed validation data at each shape. Both layer backends load the same expert/router weights and inputs. Synthetic distributions use a controlled diagonal router: equal counts, 30% forced to expert 0 plus balanced remainder, or 90% forced plus remainder. Thus actual expert 0 shares are about 36% and 91%; exact observed group counts are in the CSV. Padding capacity and ratio are explicit. Peak allocated/reserved bytes include the common full-model/checkpoint/input-capture fixture; they are not the standalone layer footprint. Short cases cannot provide reliable GPU utilization, so that field is blank rather than fabricated.

Full-model timing compares DenseCompute, Sparse reference, Sparse grouped and current Sparse grouped+Ngram. Width320, nine layers, frozen32768 tokenizer/shards, context 1024, microbatch 2, accumulation 4, identical policy/data order, and twenty measured updates per loop are fixed. Each loads its existing Phase1A weights with a fresh AdamW; this is speed timing, not continued quality training. The three warm updates precede measurement. Separate six-repeat forward and forward+backward latency calls use the same batch and warmed CUDA.

The pure-step loop reuses four preloaded identical batches. The following distinct wall loop reads batches and flushes JSONL logs. Both include four forward/backward microbatches, finite checks, scalar LM-loss observations, clipping, ordinary AdamW policy, scheduler and explicit synchronization. Wall excludes initialization, evaluation and checkpoint I/O. Consequently its rate can occasionally exceed pure-step rate due to short-run timing variation and a distinct later trajectory. It cannot replace the historical Phase1A wall denominator, which included evaluation/checkpoint costs. Each loop consumes 163840 tokens/model/repetition. Two repetitions reverse model order; aggregate rate is summed tokens / summed seconds, not a quality score or a confidence interval. Individual ranges are retained.

Geometry compares grouped 12x160, 6x320 and 4x480 at equal routed stored FFN capacity: E*FFN=1920 in three width 320 capacity layers, giving 5529600 routed weights. Router weights and active per-token FFN compute differ; neither active compute nor total parameters is forced equal. All variants freshly initialize with seed42, same data and fresh optimizer, ten updates per separate pure/wall loop, two reverse-order repetitions. First-to-last warm/pure-loop LM loss is a finite learning smoke only. No old trained12-versus-fresh6 comparison is used. The base twelve-expert research architecture is not automatically changed by a speed winner.

Current N-gram timing uses its existing trained memory checkpoint and unchanged banks/lookup/gate. Its architecture-specific weights can give slightly different natural routing; this is incremental whole-model runtime cost, not a controlled same-output quality ablation. No table capacity sweep or redesign occurs. Single endpoint GPU-utilization samples are reported as coarse samples, not occupancy or average utilization.

`results/moe_benchmark_provenance.json` binds core/config/tool/test bytes, tokenizer/data manifest and binary shards, hardware/software and final CSV hashes. Large checkpoint identity is recorded with size/SHA256/origin in the Prompt-1 archive manifest. Runtime configs cap a direct short training invocation at 327680 tokens; the benchmark tool independently enforces 20-step (geometry10) loops. Original Phase1A configs remain intact.

Reproduce sequentially, after correctness:

```powershell
.\.venv\Scripts\python.exe -m tools.moe_benchmark micro
.\.venv\Scripts\python.exe -m tools.moe_benchmark full
.\.venv\Scripts\python.exe -m tools.moe_benchmark granularity
.\.venv\Scripts\python.exe -m tools.moe_reference_profile --grouped
.\.venv\Scripts\python.exe -m tools.moe_report
```

These commands do not launch the next quality phase. Historical leaderboard rows retain their original scope; runtime tables are separate rather than mixing short speed-only loops into that quality-training board.

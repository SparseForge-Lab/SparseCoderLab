# Implementation correctness, memory and scaling review

The implementation now supports bounded training outputs, explicit AdamW decay groups, CPU checkpoint staging and larger streamed shards. RTX 5070 measurements reduced peak allocated memory by 36â€“42% at the existing context/batch geometry. Short throughput measurements show changes between approximately -1.8% and +1.5%; they do not establish a general speedup. Historical research results, the frozen multilingual pilot and canonical checkpoints remain unchanged.

## Confirmed issues and disposition

| Area | Confirmed problem | Change and remaining limit |
|---|---|---|
| AdamW | A single decay group included biases, normalization vectors, the nonzero Ngram gate bias and lookup tables. Zero-gradient AdamW could move the gate bias toward zero. | Two named groups decay matrices except Ngram tables; all vectors/biases/tables receive zero decay. Tests prove exact, disjoint, exhaustive membership and the gate-bias behavior. `legacy_all` preserves the historical decay recipe when explicitly selected. |
| Ngram state | Embedding gradients and Adam moments cover full lookup tables. | Dense gradients/moments remain. Sparse or row-wise updates would change inactive-row momentum, global clipping and resume semantics; no compatible equivalent was established. Table decay is removed under the new policy. |
| Grouped MoE | Each forward stacked expert parameters, read CUDA counts on the host and allocated padded buffers from maximum load. | Persistent contiguous banks back the same named expert Parameters; gradients go directly to individual experts, with `None` for inactive experts. A separate optimizer active-flag host read is removed. The single allocation-time count read and padded batched GEMMs remain. A variable-size candidate failed the declared BF16 gradient tolerance and was removed. |
| Training output / CE | Ordinary training returned full logits/hidden states and built a full FP32 CE tensor. | Explicit loss-only calls use checkpointed linear/CE chunks (default 1,024 tokens), retain minimal loss metrics and release references after backward. The public forward defaults remain compatible for callers requesting full outputs. |
| Checkpoints | Some full-state loads mapped complete payloads to CUDA, temporarily duplicating state. | Loads stage on CPU; model/optimizer loading transfers live tensors. Evaluation loads remain on CPU until needed. Full resume verifies model, Adam, scheduler, counters and RNG. |
| Attention | GQA physically repeated K/V; RoPE rebuilt trigonometric tensors each forward. | Nonpersistent FP32 RoPE caches grow safely with length/device/dtype. Native GQA is tested and opt-in; repeated K/V remains the default pending stronger whole-model measurements. |
| Timing / CPU memory | Main training and profiler retained unbounded per-step timing lists. | Streaming counts/maxima and at most ten profiler samples. The profiler now follows the decay groups and minimal-output path while preserving its constant learning rate. |
| Shards | Lookup scanned shard lengths linearly and opened all mmaps; small shards would create many files. | Binary-search lookup, lazy LRU of eight maps and copied slices safe across eviction; boundary/empty/negative slices tested. New default shard size is 134,217,728 uint16 tokens (256 MiB); explicit historical sizes remain honored. |
| Large-file I/O | Several hashes read complete corpus/shard files; packing could duplicate monolithic token payloads. | Shared streaming SHA-256 and direct physical shard writes with bounded buffers; the legacy packer spools token payloads to disk, and conversion reads in chunks. Corpus metadata/body records and some small config/manifest reads remain in memory. |
| Dedup | Per-document shingle sets stayed in memory; Python MinHash arithmetic was costly; short files bypassed near matching. | Shingles spool to disk with an eight-item LRU; vectorized Mersenne arithmetic exactly matches the Python signatures. Short Python files can use identical normalized ASTs; strings/names/literals remain significant. Other short files keep exact-only handling. Separate exact/near/short counts and conservative false-positive tests. Body records and LSH metadata remain in memory. |
| Packing / budget | Packed unrelated documents could share attention/Ngram history, and sorted early truncation biased later repositories/languages. | New repository manifests select `document_isolated_v1`: same-document causal attention, reset Ngram history and ignored cross-document targets. SQLite-backed proportional token strata with seeded hash ranking replace prefix truncation in the new preparation path. Whole-document rounding can leave strata absent and is reported. Existing legacy manifests keep ordering/cursors; legacy packer quotas remain a limitation. |
| Evaluation | First-N files and prefix-only windows biased samples. | Bounded deterministic hash sampling over documents and positions; comparison manifests record version/context/microbatch identity. Loss aggregation weights valid prediction counts. Historical evaluation indices remain intact. |
| FIM | Character cuts could produce tiny or token-pathological spans. | Token-aware deterministic selection requires at least 16 independently encoded tokens in each span, with bounded attempts and Unicode-safe fallback; unsuitable files are skipped. Original source reconstructs exactly. Historical character-based API remains available for compatibility. |
| Diagnostics | Allocated/reserved memory and component/I/O times were insufficiently separated. | Training exposes data wait, optimizer timing and current/peak allocated/reserved bytes. A bounded diagnostic captures attention, router, Ngram, dispatch and checkpoint times with explicit attribution limits. |

## RTX 5070 before/after measurements

[`rtx5070_v3.json`](../results/implementation_fixes/rtx5070_v3.json) compares baseline commit `06f0ef1f5f9cf4ce58e9073f1ff845c8ac27df1d` with the revised implementation. Each variant uses read-only canonical 100M weights, synthetic inputs, context 1,024, microbatch 8, accumulation 1, BF16, TF32, three warmups and six measured updates. Native GQA is disabled. Timing includes backward, clipping, optimizer, scheduler and synchronization; it excludes data loading, initialization and checkpoint I/O. Adam moments are initialized for this benchmark, rather than inherited from a historical full-state run. Before/after also changes the documented decay policy, so later loss trajectories are not expected to be identical.

| Model | Before tokens/s | After tokens/s | Peak allocated GiB before â†’ after | Peak reserved GiB before â†’ after |
|---|---:|---:|---:|---:|
| Dense | 47,407.52 | 48,134.55 | 7.31 â†’ 4.22 | 7.53 â†’ 4.66 |
| Sparse | 40,085.06 | 39,403.40 | 8.12 â†’ 5.03 | 8.40 â†’ 5.59 |
| Sparse + 10M Ngram | 39,275.30 | 38,577.16 | 8.28 â†’ 5.19 | 8.58 â†’ 5.77 |
| Sparse + 25M Ngram | 37,918.83 | 38,092.07 | 8.50 â†’ 5.41 | 8.77 â†’ 5.92 |

The initial 1Ã—32-token probe had identical logits, loss, routes and gradients for all variants. Multi-chunk loss/gradient parity is separately tested with and without activation checkpointing on CPU and CUDA. BF16 reduction ordering and corrected decay can change subsequent updates. The short post-warmup allocation traces show allocator variation and cached reservation; they provide no evidence of a sustained leak and cannot exclude a longer-run leak. Reserved memory is allocator capacity, distinct from live tensors and temporary peaks. Dense gradients and two full Adam moment tensors are expected persistent Ngram costs.

Isolated grouped MoE measurements at batch 8/context 1,024 retained exact outputs/input/parameter gradients and routes for balanced and 90%-skewed distributions. Balanced throughput was 1,494,890.51 â†’ 1,419,240.83 tokens/s and skewed throughput 673,742.39 â†’ 666,474.12. Peak allocated bytes were 277,090,816 â†’ 277,680,640 (balanced) and 757,865,472 â†’ 733,486,080 (skewed); reserved bytes were 320,864,256 â†’ 325,058,560 and 889,192,448 â†’ 813,694,976. There is no isolated MoE speedup claim.

The isolated native-GQA check measured forward/backward 3.417 â†’ 3.319 ms and peak allocation 200,296,448 â†’ 194,004,992 bytes; reserved bytes were 232,783,872 for both. Output difference was zero and maximum gradient difference 9.313e-10. One shape and a short benchmark do not justify changing the whole-model default.

CPU MinHash signatures for 10,000 synthetic shingle values measured median 143.448 â†’ 6.787 ms over seven repetitions (21.13Ã—), with identical signatures. This is a signature microbenchmark, not corpus throughput or dedup recall.

## Resume and numerical compatibility

Parameter names and model state-dict shapes remain unchanged. Expert banks and RoPE caches are nonpersistent; they add no checkpoint keys. Old single-group Adam state migrates by validated named-parameter order into the two groups, retaining moments, steps and learning rate. Scheduler group lists are replicated consistently. Invalid counts, identities or moment shapes fail explicitly. Corrected decay changes future optimization; migration is state preserving but does not promise the old training trajectory. An explicit `legacy_all` policy keeps the old decay recipe.

Boundary isolation changes the training objective for new repository manifests, including attention, Ngram hashing and ignored cross-document labels. Frozen legacy streams preserve order/cursor semantics. Boundary-aware packing with enabled MTP raises an explicit unsupported-combination error; that combination has no validated boundary-safe draft-target implementation.

The diagnostic full-state checkpoint was 1,215,230,659 bytes: save 3.782 s, CPU-staged resume 0.483 s. Model/Adam/scheduler hashes and counters matched; separate CPU/CUDA tests verify persisted RNG. This is one warmed local-disk measurement, not a general checkpoint speedup.

The component diagnostic was one instrumented 25M-Ngram step after two warmups (209.988 ms). CUDA intervals: attention 18.865 ms, Ngram 1.899 ms, router linear 0.195 ms, MoE dispatch 10.521 ms and optimizer 39.770 ms. These cover forward components and optimizer, include submission gaps, exclude attributed backward kernels and use preloaded inputs. They are not an additive kernel-time breakdown. Production metrics separately report dataloader wait.

## Validation and evidence identity

The CUDA-inclusive release suite passed **106 tests in 30.60 s**, including 12 GPU tests. After the final profiler-only change, CPU publication validation passed **95 tests in 28.66 s**, with 12 GPU tests deselected and CUDA devices hidden. This adds one CPU profiler regression test. The GPU suite was completed before the subsequent GPU pause; no further CUDA jobs were launched during that pause.

Coverage includes exhaustive optimizer groups, nondecaying biases, dense Ngram gradients, minimal/full loss and gradients, activation checkpointing, CPU load staging, full resume state/RNG, old-group migration, GQA/RoPE, persistent MoE bank views/deepcopy, balanced/skew/empty expert dispatch, LRU boundaries, document isolation/Ngram resets/cursors, exact streaming hashes/shard conversion, stratified budgets/validation windows, MinHash arithmetic, short-file false positives and token-aware FIM reconstruction. All reference functional tasks and sandbox limits are exercised by CPU integration tests.

The current-source FP32 CPU parity report includes five updates from a trusted Phase-1A full-state checkpoint. Earlier deterministic BF16 CUDA five-update parity is retained as versioned evidence; its source hashes differ from final sources. The accepted full-model and isolated MoE measurements plus the release GPU tests cover the revised implementation; the older report is not relabeled as final-source evidence. [`validation_v1.json`](../results/implementation_fixes/validation_v1.json) records final file hashes, exact test results and report binding/mismatch information. Measurements remain immutable when later comments, validation bookkeeping or evaluation code change.

```powershell
python -m pytest -q
# CPU-only publication checks while CUDA use is paused:
$env:CUDA_VISIBLE_DEVICES = ''
python -m pytest -q -m "not gpu"
# Only when GPU work is authorized; choose fresh outputs:
python -m tools.implementation_benchmark --help
python -m tools.implementation_diagnostics --output results/implementation_fixes/diagnostics_v2.json
python -m tools.moe_parity --deterministic --output results/implementation_fixes/moe_parity_v2.json
```

Remaining priorities are validated variable-size dispatch without changed BF16 routing/gradients, a mathematically compatible sparse optimizer, full-corpus metadata scaling, independent held-out language/repository coverage and broader contamination tests. This implementation review does not authorize or complete a new training phase.

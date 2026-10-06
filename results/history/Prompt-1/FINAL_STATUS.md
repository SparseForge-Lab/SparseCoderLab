# FINAL_STATUS — Phase 1A review stop, 2026-10-05

**Completed:** three fresh matched synthetic engineering runs and trained-checkpoint memory ablation. **Stopped:** research corpus acquisition, Phase 1B and later experiments, under the explicit slowdown review condition.

| Model | Validation NLL | Code NLL | General NLL | Bits/token | Wall s | Wall tok/s | Step tok/s | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DenseCompute | 4.349110 | 3.984163 | 5.071468 | 6.274439 | 184.80 | 27085 | 30324 | 1.476 |
| SparseV3 | 4.345538 | 3.980094 | 5.073375 | 6.269287 | 492.90 | 10155 | 10874 | 1.566 |
| SparseV3Ngram | 4.349094 | 3.982275 | 5.079932 | 6.274416 | 527.25 | 9493 | 10174 | 1.636 |

All models consumed exactly 5,005,312 tokens, seed 42. All 32 regression tests passed after measurement changes; the current-source gate is retained in results/phase0_gate.json. Original Phase 0 evidence and byte-exact experiment source are archived under results/history. Original configs, tokenizer and corpus remain intact.

1. **Did SparseV3 survive?** Stable training survived; promotion did not. Its -0.003571 synthetic NLL gap is too small and inconsistent to establish a quality advantage.
2. **Did N-gram help?** Aggregate benefit is unestablished. Same-checkpoint residual ablation increases mixed NLL by only 0.00005400; the small code-subset effect is not reproduced across seeds or real data.
3. **Fairest dense baseline?** DenseCompute for the nearly matched active estimate. DenseSize is an unrun optional stored-capacity reference.
4. **Expert specialization/locality?** Unknown. Route balance recovered, but representative held-out trace/packing studies were stopped before Part E.
5. **Measured speed penalty?** SparseV3 2.67x and Ngram 2.85x wall time at equal tokens; training-step ratios 2.79x/2.98x. These are observed timings, not large-model projections.
6. **Unproven?** Real coding quality, corpus/tokenizer suitability, real-data architecture ranking, seed reproducibility, memory-capacity benefit, useful locality/offload, MTP, RouteAhead and compaction.
7. **Next experiment?** Architecture redesign/review, focused on dispatch overhead and memory contribution, before resuming a verified real-data comparison. No next phase started automatically.

The separate Documentation files 23–27 explain methods, measurements and every omitted phase. results/phase1a_comparison.csv and phase1a_curves.csv contain measured training evidence with declared calculated-estimate fields. phase1b_comparison.csv explicitly contains not-run statuses and blank measurements. Research lane directories contain readiness notes only; no real corpus or new tokenizer exists.

Loss/throughput/gates/gradients come from actual execution; parameter counts, active/FLOP estimates, bits/token, CV and ratios are calculated from those observations. Across-seed and real-data conclusions remain unknown. The validation fixture is repetitive and its engineering losses must not be marketed as coding quality. See results/phase1_review_decision.json for the stop decision and 23_phase1a_protocol.md for the predeclared endpoint.

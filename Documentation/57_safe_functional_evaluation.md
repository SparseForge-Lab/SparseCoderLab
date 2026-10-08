# Bounded functional evaluation of saved Python completions

The saved 100M-token generation screen now has a completed functional report: [`functional_100M_v3.json`](../results/research_v2_real/functional_100M_v3.json). It scores 700 existing completions: four model variants, seven temperatures and 25 elementary Python tasks with 94 fixed cases. It does not generate new answers or change checkpoints.

The evaluator concatenates the original prompt and raw completion without repair, truncation or extraction. Host-side checks compare returned values and captured stdout with the fixed test oracle. Booleans are distinct from numbers; numerical comparison uses explicit tolerance and rejects nonfinite or overflowing conversions. Loop constraints, Python parseability and repetition are separate measurements.

| Model | Greedy correct / 25 | Parseable | Repetition heuristic | Correct without repetition |
|---|---:|---:|---:|---:|
| Dense | 2 | 28% | 100% | 0 |
| Sparse | 0 | 32% | 100% | 0 |
| Sparse + 10M Ngram | 0 | 52% | 100% | 0 |
| Sparse + 25M Ngram | 0 | 28% | 100% | 0 |

These are finite toy-task results, not pass@k, repository engineering, instruction following or a model ranking. The completion records declare CUDA generation, despite the historical input filename containing `cpu`. They include model identities and token counts but omit checkpoint SHA-256 at generation time. The canonical checkpoint references in the report cannot retrospectively prove which weights produced those answers.

## Execution boundary

Generated code executes in pinned CPython 3.14.7 inside Wasmtime 49.0.0. Installation validates archive/wheel hashes and produces a local runtime integrity receipt. The guest receives arguments but no expected answers. Only the runtime standard library is mounted read-only; no project directories, host environment, network, writable mounts or stdin are supplied.

Each fresh guest store has a 128 MiB linear-memory limit, 3 billion fuel units, a 3-second execution deadline and a 16 KiB output cap. Source input is limited to 16 KiB. Compilation occurs outside the execution deadline, and the guest memory limit does not cap total host RSS. Host AST parsing and guest execution use distinct Python runtimes. This is a bounded evaluation tool rather than a general untrusted-code hosting service.

Tests execute every reference solution and exercise filesystem traversal/write attempts, network/process attempts, environment isolation, time/fuel, memory and output limits, fresh-store recovery and oversized-source reporting. Oversized sources report zero execution time and fuel consumption. A very large integer comparison is rejected without overflowing the host scorer.

## Contamination screen and reproducibility

[`coding_overlap_pilot_v1.json`](../results/research_v2_real/coding_overlap_pilot_v1.json) records zero matches across 5,526 retained pilot documents after restoring FIM source text. The scanner covers exact task/prompt/reference fingerprints for this 25-task fixture. It does not establish semantic nonoverlap, external benchmark cleanliness or absence of exposure in earlier training. Any reviewed exclusion requires a new dataset version; the frozen pilot remains unchanged.

```powershell
python -m tools.setup_wasi_eval
python -m tools.score_saved_python --input results/prompt3_cpu_generation_benchmark.jsonl --output results/research_v2_real/functional_100M_v4.json
python -m pytest -q -m "not gpu"
```

The raw generations and runtime binaries are local artifacts excluded from publication. The public report retains completion hashes, outcomes, runtime identity, source-code hashes, decoding policy and canonical checkpoint references. Re-scoring needs the local saved generations; output paths must use a fresh report version.

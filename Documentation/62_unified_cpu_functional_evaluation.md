# Unified functional evaluation and telemetry

`configs/eval/unified_python_v1.json` freezes 100 prompts and 376 input/output cases. The fixture contains 25 underlying elementary problem families with four wordings each. Those paraphrases are correlated: report them as wording variants, not as 100 independent algorithms. The original 25-task `simple_python_v1` benchmark and all of its historical scores remain unchanged.

`tools/unified_eval.py score` evaluates saved generations only. It never loads a model or uses CUDA. Generated Python runs inside the existing hash-verified CPython/WASI guest (`src/eval/wasi_python.py`): no project mount, network, writable host directory, inherited environment, or test oracle; the guest has fuel, memory, deadline, source, and output caps. Outcomes distinguish empty response, failed extraction, syntax error, timeout, runtime failure, wrong answer, and correct answer. Reaching the generation cap is retained as an independent `truncated` flag; available code is still extracted, parsed, and executed, and a capped candidate can pass if it solves every case. Parseability, repetition, extraction mode, passed cases, captured output, throughput, latency, checkpoint identity, and runtime/source hashes are retained independently. The guest captures stdout and stderr separately for module setup and each test case, including partial streams when a candidate raises; raw WASI stderr is retained as well. pass@1 is reported per observed task. Missing samples are excluded from pass@k estimates.

The scorer accepts benchmark fixtures with 100–300 unique task IDs. `unified_python_v1` is the checked-in default. It refuses to overwrite either report file and verifies that inputs did not change during scoring. The text report is a compact summary; JSON keeps each completion and its provenance. The explicit `sum_values` loop requirement is reported separately from output correctness, following the earlier scorer's distinction between functional behavior and implementation constraint.

```powershell
python -m tools.setup_wasi_eval
# Run only after the matched checkpoint campaign and its final gate have finished.
python -m tools.generate_unified_eval --output results/unified_python_250M_greedy_r1.jsonl
python -m tools.unified_eval score `
  --benchmark configs/eval/unified_python_v1.json `
  --generations results/unified_python_250M_greedy_r1.jsonl `
  --efficiency results/unified_python_250M_greedy_r1.efficiency.jsonl `
  --expected-variants dense75_ref sparse75 sparse75_ngram10m sparse75_ngram25m `
  --expected-training-tokens 250003456 `
  --output results/unified_python_250M_functional_r3.json
```

The existing generation path remains `tools/generate_canonical_python.py`; its frozen 25-prompt, seven-temperature study remains historically distinct. `tools/generate_unified_eval.py` is the explicit GPU step for the four matched 250M-token 75M-parameter variants. It validates the final manifest and each checkpoint via the existing gate loader, uses seed 42, greedy decoding, a maximum of 128 new tokens, and batch size four by default, and checks the user GPU-pause marker between batches. It refuses to overwrite generations or efficiency receipts. Do not run it while the training campaign or its final gate is active. Each record binds prompt, decoding settings, checkpoint SHA-256 and size, token count, benchmark/tokenizer/sampler identities, model-load and generation timing, CPU RSS, and peak VRAM. The scorer preserves `null` for unavailable values.

`telemetry` aggregates Top-1 or Top-k selected-expert traces by category, language, and source type. It reports load counts, entropy, coefficient of variation, min/max share, unused experts when the configured count is supplied, over-2x-uniform load, and transitions. It can also carry forward the existing repository-gate router and N-gram summaries with their source hash. Full confidence margins need router probabilities/logits and are explicitly unavailable from selected-ID traces. Sequence helpers report switching, run lengths, transition counts/matrices, and local/global transition entropy. The current production router is not changed; trace collection remains opt-in and should sample rather than retain every token.

N-gram diagnostics accept existing bounded `NgramMemory.statistics()` output, including lookup-frequency percentiles derived from its histogram. They explicitly retain sample scope; unavailable statistics are not zero. `evaluation_memory_ablation()` requires evaluation mode and restores the exact prior ablation flag even after failure. It supports same-checkpoint residual-off evaluation; it does not replace comparisons between separately trained Sparse and N-gram models.

```powershell
python -m tools.unified_eval telemetry --routes results/routes_sample.jsonl `
  --ngram results/ngram_statistics.json --output results/unified_telemetry_r1.json
python -m tools.unified_eval architecture --vocab 32000 --d-model 2048 --layers 24 `
  --dense-ffn 5632 --experts 8 --top-k 2 --moe-layers 8
python -m tools.unified_eval queue --root experiments --output results/checkpoint_eval_queue_r1.json
```

Checkpoint discovery is queue-only and records SHA-256 values. It never starts evaluation, reads model weights into memory, or requests the GPU. Existing hashes can be passed with `--existing` to deduplicate into a fresh output file. Historical report imports copy the original JSON into a new, versioned wrapper with its source hash, Prompt label, and an explicit `exact`, `near_match`, or `unmatched` classification; they never rewrite source reports or claim continuous learning from Prompt-3 to Prompt-5.

The architecture command reports first-order stored and active parameter estimates, weight bytes, and optimizer-state bytes. For dense reference sizing, try the following approximate 750M, 7.5B, and 75B shapes (vocabulary 32K); use expert and n-gram options to model alternate storage/active-count assumptions:

```powershell
python -m tools.unified_eval architecture --vocab 32000 --d-model 1536 --layers 32 --dense-ffn 4096
python -m tools.unified_eval architecture --vocab 32000 --d-model 4096 --layers 48 --dense-ffn 11008
python -m tools.unified_eval architecture --vocab 32000 --d-model 12288 --layers 52 --dense-ffn 32768
```

These estimates omit activations, attention implementation details, fragmentation, optimizer variants, and several small parameter groups. They are planning calculations, not training-memory guarantees or model-quality comparisons. None of these reports produces a composite score or declares an architecture winner.

Validation:

```powershell
python -m pytest -q tests/test_unified_eval.py tests/test_wasi_eval.py
python -m pytest -q -m "not gpu"
```

The first command runs the CPU helper regressions plus the existing sandbox suite when the pinned runtime is installed; otherwise runtime-dependent tests skip. These tests do not load a checkpoint or run a GPU model evaluation.

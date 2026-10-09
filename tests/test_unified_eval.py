"""CPU unit tests for unified evaluation aggregation and planning helpers."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from src.eval.unified import (architecture_estimate, checkpoint_candidates,
                              classify_completion, evaluation_memory_ablation,
                              extract_code, ngram_summary, pass_at_k,
                              routing_summary, sequence_metrics,
                              summarize_completions)
from src.eval.wasi_python import WasiPython
from tools.score_saved_python import score
from tools.unified_eval import (deduplicate_candidates, structural_requirements,
                                validate_generation_matrix, validate_policy_consistency)


class Runtime:
    def execute(self, source, function, arguments):
        value = 999 if "999" in source else None
        return {"kind": "completed", "results": [{"value": value if value is not None else a[0] + a[1], "stdout": ""} for a in arguments],
                "wall_seconds": .001, "fuel_consumed": 5}


class TimeoutRuntime:
    def execute(self, source, function, arguments):
        return {"kind": "timeout", "wall_seconds": 3.0, "fuel_consumed": 100, "stderr": ""}


def task():
    return {"id": "sum", "function": "add", "cases": [{"args": [2, 3], "expected": 5, "stdout": ""}]}


def test_extract_and_classify_parseability_is_not_correctness():
    code, mode = extract_code("```python\ndef add(a,b): return a+b\n```")
    assert mode == "fenced" and "def add" in code
    result = classify_completion("def add(a,b): return 999", "", task(), Runtime())
    assert result["parseable"] and result["outcome"] == "wrong" and not result["functional_correct"]
    assert classify_completion("", "", task(), Runtime())["outcome"] == "empty"
    assert classify_completion("Here is the solution", "", task(), Runtime())["outcome"] == "extraction_failure"
    assert classify_completion("def add(:", "", task(), Runtime())["outcome"] == "syntax_error"
    body = classify_completion("return a+b", "def add(a,b):", task(), Runtime())
    assert body["outcome"] == "correct" and body["extracted_code"] == "return a+b"
    assert classify_completion("def add(a,b): return a+b", "", task(), TimeoutRuntime())["outcome"] == "timeout"
    assert classify_completion("def add(a,b): return a+b", "", task(), Runtime(),
                               token_limit=10, generated_tokens=10)["outcome"] == "truncated"


def test_classification_preserves_module_and_per_case_output_streams():
    class CaptureRuntime:
        def execute(self, source, function, arguments):
            return {"kind": "completed", "module_stdout": "module out\n",
                    "module_stderr": "module err\n",
                    "results": [{"value": 5, "stdout": "call out\n", "stderr": "call err\n"}],
                    "wall_seconds": .002, "fuel_consumed": 7}

    captured_task = {"id": "sum", "function": "add", "cases": [
        {"args": [2, 3], "expected": 5, "stdout": "call out\n"}]}
    result = classify_completion("def add(a,b): return a+b", "", captured_task, CaptureRuntime())
    assert result["functional_correct"]
    assert result["execution_stdout"] == "module out\n"
    assert result["execution_stderr"] == "module err\n"
    assert result["observations"][0]["stdout"] == "call out\n"
    assert result["observations"][0]["stderr"] == "call err\n"

    class PartialErrorRuntime:
        def execute(self, source, function, arguments):
            return {"kind": "runtime_error", "exception": "RuntimeError",
                    "module_stdout": "module out\n", "module_stderr": "module err\n",
                    "partial_stdout": "before failure\n", "partial_stderr": "candidate err\n",
                    "stderr": "wasm err\n", "wall_seconds": .002, "fuel_consumed": 7}

    failed = classify_completion("def add(a,b): return a+b", "", captured_task, PartialErrorRuntime())
    assert failed["execution_stdout"] == "module out\n"
    assert failed["execution_stderr"] == "module err\nwasm err\n"
    assert failed["partial_stdout"] == "before failure\n"
    assert failed["partial_stderr"] == "candidate err\n"


def test_pass_at_k_excludes_insufficient_draws_and_summary_keeps_degeneration_separate():
    rows = [{"task_id": "a", "functional_correct": True, "outcome": "correct", "parseable": True,
             "repetition": True},
            {"task_id": "a", "functional_correct": False, "outcome": "wrong", "parseable": True,
             "repetition": True},
            {"task_id": "b", "functional_correct": False, "outcome": "wrong", "parseable": False,
             "repetition": False}]
    assert pass_at_k(rows, 1)["estimate"] == .25
    assert pass_at_k(rows, 2) == {"k": 2, "estimate": 1.0, "tasks_scored": 1,
                                   "tasks_with_too_few_samples": 1, "scope": pass_at_k(rows, 2)["scope"]}
    summary = summarize_completions(rows)
    assert summary["functional_accuracy"] == 1 / 3 and summary["repetition_rate"] == 2 / 3


def test_route_and_sequence_metrics_include_transitions_and_configured_unused_experts():
    seq = sequence_metrics([[0, 0, 1, 1]], expert_count=3)
    assert seq["switches"] == 1 and seq["switch_rate"] == 1 / 3
    assert seq["run_length_mean"] == 2 and seq["transition_matrix"][0][1] == 1
    result = routing_summary([{"category": "code", "language": "Python", "source_type": "repo",
                               "expert_count": 3, "layers": {"0": [0, 0, 1]}},
                              {"category": "code", "language": "Python", "source_type": "repo",
                               "expert_count": 3, "layers": {"0": [1, 2]}}])
    layer = result["groups"]["code/Python/repo/layer0"]
    assert layer["unused_experts"] == 0 and layer["full_expert_count"] == 3
    assert layer["transitions"]["0->1"] == 1
    assert layer["confidence_margin"] is None
    top_k = routing_summary([{"category": "code", "language": "Rust", "source_type": "repo",
                              "expert_count": 4, "layers": {"0": [[0, 2], [1, 3]]}}])
    assert top_k["groups"]["code/Rust/repo/layer0"]["top_k_slots_per_token"] == 2
    assert top_k["groups"]["code/Rust/repo/layer0"]["counts"] == {0: 1, 1: 1, 2: 1, 3: 1}


def test_ngram_percentiles_and_missing_is_not_zero():
    assert ngram_summary(None)["available"] is False
    result = ngram_summary({"banks": [{"order": 2, "rows": 10, "utilized_rows": 3,
                                        "distinct_ngrams": 4, "collisions": 1,
                                        "collision_fraction": .25,
                                        "lookup_frequency_histogram": {"1": 2, "3": 1}}]})
    assert result["available"] and result["banks"][0]["lookup_frequency_percentiles"]["p50"] == 1


def test_ablation_restores_prior_state_on_success_and_error():
    memory = SimpleNamespace(ablate=False, training=False)
    with evaluation_memory_ablation(memory):
        assert memory.ablate
    assert not memory.ablate
    memory.ablate = True
    with pytest.raises(RuntimeError):
        with evaluation_memory_ablation(memory):
            raise RuntimeError("test")
    assert memory.ablate is True


def test_queue_is_manual_only_hash_deduplicable_and_architecture_estimates_are_monotonic(tmp_path: Path):
    folder = tmp_path / "checkpoints"; folder.mkdir()
    (folder / "model_250M.pt").write_bytes(b"small fixture")
    (folder / "other.pt").write_bytes(b"ignored")
    queued = checkpoint_candidates(tmp_path)
    assert len(queued) == 1 and queued[0]["status"] == "queued_manual_review" and not queued[0]["gpu_used"]
    small = architecture_estimate(vocab=100, d_model=32, layers=2, dense_ffn=64)
    large = architecture_estimate(vocab=100, d_model=64, layers=2, dense_ffn=128)
    assert large["stored_parameters_estimate"] > small["stored_parameters_estimate"]
    assert large["active_parameters_per_token_estimate"] > small["active_parameters_per_token_estimate"]
    with pytest.raises(ValueError):
        architecture_estimate(vocab=100, d_model=32, layers=2, dense_ffn=64, experts=1, top_k=2)


def test_frozen_unified_fixture_has_100_prompts_and_25_correlated_families():
    benchmark = json.loads(Path("configs/eval/unified_python_v1.json").read_text(encoding="utf8"))
    assert len(benchmark["tasks"]) == 100
    assert len({task["id"] for task in benchmark["tasks"]}) == 100
    assert len({task["family_id"] for task in benchmark["tasks"]}) == 25
    assert sum(len(task["cases"]) for task in benchmark["tasks"]) == 376
    assert "correlated" in benchmark["scope"]


def test_generation_matrix_requires_every_variant_task_once():
    tasks = [{"prompt_number": 1}, {"prompt_number": 2}]
    rows = [{"variant": variant, "temperature": 0.0, "prompt_number": number}
            for variant in ("dense", "sparse") for number in (1, 2)]
    assert len(validate_generation_matrix(rows, tasks, ["dense", "sparse"])) == 4
    with pytest.raises(ValueError, match="incomplete"):
        validate_generation_matrix(rows[:-1], tasks, ["dense", "sparse"])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_generation_matrix(rows + [rows[0]], tasks)
    candidates = [{"sha256": "a", "path": "first"}, {"sha256": "a", "path": "same-bytes"},
                 {"sha256": "b", "path": "different"}]
    assert [row["path"] for row in deduplicate_candidates(candidates, set())] == ["first", "different"]
    assert deduplicate_candidates(candidates, {"a"}) == [candidates[2]]
    sum_task = {"id": "sum_values_wording1", "family_id": "sum_values"}
    assert structural_requirements(sum_task, "def sum_values(values):", "def sum_values(values):\n    return sum(values)") == {
        "loop_constraint_required": True, "uses_loop": False, "loop_constraint_met": False}
    assert structural_requirements(sum_task, "def sum_values(values):", "def sum_values(values):\n    for x in values: pass") == {
        "loop_constraint_required": True, "uses_loop": True, "loop_constraint_met": True}


def test_matched_generation_policy_is_frozen_and_consistent():
    row = {"decoding": "greedy", "deterministic_algorithms": True, "gpu_name": "NVIDIA GeForce RTX 5070", "generation_seed": 42,
           "max_new_tokens": 128, "top_p": None, "top_k": None, "repetition_penalty": 1.0,
           "stop_tokens": ["<eos>", "<doc>"], "device": "cuda", "bf16": True, "tf32": True,
           "generation_batch_size": 4, "temperature": 0.0}
    assert validate_policy_consistency([row], canonical_greedy=True)["decoding"] == "greedy"
    with pytest.raises(ValueError, match="policy differs"):
        validate_policy_consistency([row, {**row, "generation_seed": 7}])
    with pytest.raises(ValueError, match="frozen deterministic"):
        validate_policy_consistency([{**row, "generation_batch_size": 2}], canonical_greedy=True)


@pytest.fixture(scope="module")
def wasi_runtime():
    if not Path(".local/wasi_eval/integrity.json").exists():
        pytest.skip("Install pinned CPU runtime with python -m tools.setup_wasi_eval")
    return WasiPython()


def test_all_100_frozen_reference_solutions_pass_the_guest(wasi_runtime):
    benchmark = json.loads(Path("configs/eval/unified_python_v1.json").read_text(encoding="utf8"))
    for case in benchmark["tasks"]:
        result = score(wasi_runtime, case["reference"], case)
        assert result["outcome"] == "correct", (case["id"], result)

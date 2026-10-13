"""Executable, matched scorer for the 500M Prompt-5 diagnostic generations.

Run only after the Dense, +10M, and +25M 500M milestone reports exist. Scoring
uses the project's pinned CPU WASI runtime; it does not load a model or CUDA.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer

from src.eval.target_function import extract_target_function, score_target
from src.eval.wasi_python import WasiPython, file_hash
from tools.score_saved_python import equivalent, repetition

TOKENS_500M = 499_998_720
VARIANTS = ("dense75_ref", "sparse75_ngram10m", "sparse75_ngram25m")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_result(runtime: WasiPython, source: str | None, task: dict[str, Any]) -> dict[str, Any]:
    if not source:
        return {"outcome": "INCOMPLETE", "passed_cases": 0, "case_count": len(task["cases"]),
                "functional_correct": False, "exception": None}
    try:
        ast.parse(source)
    except (SyntaxError, ValueError):
        return {"outcome": "SYNTAX_ERROR", "passed_cases": 0, "case_count": len(task["cases"]),
                "functional_correct": False, "exception": "SyntaxError"}
    execution = runtime.execute(source, task["function"], [case["args"] for case in task["cases"]])
    kind = execution.get("kind")
    if kind == "timeout":
        outcome = "TIMEOUT"
    elif kind == "syntax_error":
        outcome = "SYNTAX_ERROR"
    elif kind != "completed":
        outcome = "RUNTIME_ERROR"
    else:
        observations = execution.get("results", [])
        passed = sum(index < len(observations)
                     and isinstance(observations[index], dict)
                     and equivalent(observations[index].get("value"), case["expected"])
                     and observations[index].get("stdout") == case.get("stdout", "")
                     for index, case in enumerate(task["cases"]))
        complete = len(observations) == len(task["cases"])
        module_stdout = execution.get("module_stdout", "")
        module_stderr = execution.get("module_stderr", "")
        if complete and passed == len(task["cases"]) and not module_stdout and not module_stderr:
            outcome = "STRICT_FULL_OUTPUT_PASS"
        elif complete and passed == len(task["cases"]):
            outcome = "SCOPE_VIOLATION"
        else:
            outcome = "WRONG_ANSWER"
        return {"outcome": outcome, "passed_cases": passed, "case_count": len(task["cases"]),
                "functional_correct": outcome == "STRICT_FULL_OUTPUT_PASS",
                "exception": execution.get("exception"), "execution_seconds": execution.get("wall_seconds"),
                "module_stdout": module_stdout, "module_stderr": module_stderr}
    return {"outcome": outcome, "passed_cases": 0, "case_count": len(task["cases"]),
            "functional_correct": False, "exception": execution.get("exception"),
            "execution_seconds": execution.get("wall_seconds")}


def _constraint_checks(runtime: WasiPython, target_code: str | None, task: dict[str, Any],
                       target_score: dict[str, Any]) -> dict[str, Any]:
    prompt = task["prompt"].casefold()
    checks: dict[str, Any] = {}
    if target_code is None:
        return checks
    try:
        tree = ast.parse(target_code)
    except (SyntaxError, ValueError):
        return checks
    if "using a loop" in prompt:
        node = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == task["function"]), None)
        uses_loop = bool(node and any(isinstance(n, (ast.For, ast.AsyncFor, ast.While)) for n in ast.walk(node)))
        checks["uses_loop"] = {"passed": uses_loop, "required": True}
    if "new list" in prompt:
        wrapper = target_code + f"\n\ndef __forgebuilder_check(values):\n    before = values.copy()\n    result = {task['function']}(values)\n    return [result is not values, values == before, result]\n"
        check_input = [4, -2, 4, 1]
        execution = runtime.execute(wrapper, "__forgebuilder_check", [[check_input]])
        value = None
        if execution.get("kind") == "completed" and execution.get("results"):
            value = execution["results"][0].get("value")
        fresh_and_unchanged = (isinstance(value, list) and len(value) == 3
                               and value[0] is True and value[1] is True
                               and equivalent(value[2], sorted(check_input)))
        checks["returns_new_list_and_preserves_input"] = {
            "passed": fresh_and_unchanged, "required": True,
            "execution_kind": execution.get("kind"), "observed": value}
    if "or none if it is empty" in prompt:
        empty_indices = [index for index, case in enumerate(task["cases"])
                         if case["args"] and case["args"][0] == []]
        observations = target_score.get("target_observations", [])
        empty_pass = bool(empty_indices) and all(index < len(observations)
                                                 and observations[index].get("value") is None
                                                 for index in empty_indices)
        checks["empty_input_returns_none"] = {
            "passed": bool(empty_indices) and empty_pass,
            "required": True, "empty_case_count": len(empty_indices)}
    if "preserving first occurrence order" in prompt:
        duplicate_indices = [index for index, case in enumerate(task["cases"])
                             if case["args"] and isinstance(case["args"][0], list)
                             and len(case["args"][0]) != len(set(map(repr, case["args"][0])))]
        observations = target_score.get("target_observations", [])
        duplicate_pass = bool(duplicate_indices) and all(
            index < len(observations)
            and equivalent(observations[index].get("value"), task["cases"][index]["expected"])
            for index in duplicate_indices)
        checks["duplicate_order_hidden_cases_present"] = {
            "passed": duplicate_pass, "required": True, "case_count": len(duplicate_indices)}
    return checks


def _continuation_kind(completion: str, scored: dict[str, Any], tail_repeated: bool) -> str:
    if scored.get("target_status") != "complete":
        return "no_complete_target"
    end = scored.get("target_end_char")
    tail = completion[end:] if isinstance(end, int) else ""
    if not tail.strip():
        return "none"
    if tail_repeated:
        return "repetition_after_target"
    if scored.get("syntax_damage_after_target"):
        return "syntax_damage_after_target"
    if scored.get("later_redefinition_overwrites_earlier_correct"):
        return "target_redefined_after_correct"
    if re.search(r"(?m)^\s*(?:async\s+)?def\s+", tail):
        return "additional_definition"
    return "other_trailing_content"


def _score_row(row: dict[str, Any], task: dict[str, Any], runtime: WasiPython,
               tokenizer: Tokenizer) -> dict[str, Any]:
    completion = row.get("raw_output", "")
    prompt = task["prompt"]
    target = score_target(completion, prompt, task, runtime, tokenizer=tokenizer)
    extracted = extract_target_function(completion, prompt, task)
    strict_source = extracted.get("_complete_source")
    strict = _strict_result(runtime, strict_source, task)
    constraints = _constraint_checks(runtime, target.get("target_code"), task, target)
    violations = [name for name, result in constraints.items() if result.get("required") and not result.get("passed")]
    end = target.get("target_end_char")
    tail = completion[end:] if isinstance(end, int) else ""
    tail_repeated = repetition(tail) if tail else False
    cap = int(row.get("max_new_tokens", 0))
    generated = int(row.get("generated_token_count", 0))
    if target.get("target_outcome") == "pass":
        target_outcome = "TARGET_PASS"
    else:
        target_outcome = {"wrong": "WRONG_ANSWER", "syntax_error": "SYNTAX_ERROR",
                          "runtime_error": "RUNTIME_ERROR", "timeout": "TIMEOUT",
                          "extraction_failure": "INCOMPLETE", "no_code": "INCOMPLETE"}.get(
                              target.get("target_outcome"), "INCOMPLETE")
    if violations:
        target_outcome = "EXPLICIT_CONSTRAINT_VIOLATION"
    if tail_repeated and target_outcome == "TARGET_PASS":
        continuation_type = "repetition_after_target"
    else:
        continuation_type = _continuation_kind(completion, target, tail_repeated)
    classifications = [target_outcome]
    if strict["functional_correct"]:
        classifications.append("STRICT_FULL_OUTPUT_PASS")
    elif strict["outcome"] in {"SYNTAX_ERROR", "RUNTIME_ERROR", "TIMEOUT", "SCOPE_VIOLATION"}:
        classifications.append(strict["outcome"])
    if tail_repeated:
        classifications.append("REPETITION")
    if target.get("target_status") != "complete":
        classifications.append("INCOMPLETE")
    return {
        "variant": row["variant"], "stage": row["stage"], "training_tokens": row["training_tokens"],
        "checkpoint_sha256": row["checkpoint_sha256"], "prompt_number": row["prompt_number"],
        "task_id": task["id"], "temperature": row["temperature"],
        "generated_token_count": generated, "generation_cap": cap,
        "termination_inferred": "cap_reached" if cap and generated >= cap else "stopped_before_cap",
        "target_status": target.get("target_status"), "target_function_syntactically_complete": target.get("target_status") == "complete",
        "target_function_semantically_correct": bool(target.get("target_functional_correct")),
        "target_pass_including_explicit_constraints": bool(target.get("target_functional_correct") and not violations),
        "target_outcome": target_outcome, "target_passed_cases": target.get("target_passed_cases"),
        "classifications": classifications,
        "target_case_count": target.get("target_case_count"),
        "strict_outcome": strict["outcome"], "strict_full_output_pass": strict["functional_correct"],
        "strict_passed_cases": strict["passed_cases"], "strict_case_count": strict["case_count"],
        "explicit_constraint_checks": constraints, "explicit_constraint_violations": violations,
        "natural_stop_inferred": cap > 0 and generated < cap,
        "target_tokens_after": target.get("target_tokens_after"),
        "tokens_after_target_approximate": target.get("target_tokens_after"),
        "target_complete_only_at_cap": bool(target.get("target_status") == "complete" and generated >= cap),
        "correct_target_later_ruined": bool(target.get("target_functional_correct") and not strict["functional_correct"]),
        "target_definitions": target.get("target_definitions", 0),
        "later_redefinition_overwrites_earlier_correct": bool(target.get("later_redefinition_overwrites_earlier_correct")),
        "trailing_nonblank_lines": target.get("trailing_nonblank_lines", 0),
        "repetition_after_target": bool(tail_repeated), "continuation_type": continuation_type,
        "target_code_sha256": hashlib.sha256(target["target_code"].encode()).hexdigest() if target.get("target_code") else None,
        "target_execution_seconds": target.get("target_execution_seconds"),
        "strict_execution_seconds": strict.get("execution_seconds"),
        "strict_module_stdout": strict.get("module_stdout", ""),
        "strict_module_stderr": strict.get("module_stderr", ""),
        "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
        "raw_completion": completion,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = collections.Counter(row["target_outcome"] for row in rows)
    return {
        "samples": len(rows), "target_outcomes": dict(sorted(counts.items())),
        "target_pass_count": sum(row["target_outcome"] == "TARGET_PASS" for row in rows),
        "target_pass_at_1": sum(row["target_pass_including_explicit_constraints"] for row in rows) / len(rows) if rows else None,
        "strict_full_output_pass_count": sum(row["strict_full_output_pass"] for row in rows),
        "strict_full_output_pass_rate": sum(row["strict_full_output_pass"] for row in rows) / len(rows) if rows else None,
        "target_complete_count": sum(row["target_function_syntactically_complete"] for row in rows),
        "natural_stop_count": sum(row["natural_stop_inferred"] for row in rows),
        "cap_reached_count": sum(row["termination_inferred"] == "cap_reached" for row in rows),
        "premature_stop_count": sum(row["natural_stop_inferred"] and not row["target_function_syntactically_complete"] for row in rows),
        "target_completed_with_cap_reached_count": sum(row["target_complete_only_at_cap"] for row in rows),
        "correct_target_later_ruined_count": sum(row["correct_target_later_ruined"] for row in rows),
        "continuation_after_target_count": sum(row.get("trailing_nonblank_lines", 0) > 0 for row in rows),
        "repetition_after_target_count": sum(row["repetition_after_target"] for row in rows),
        "explicit_constraint_violation_count": sum(bool(row["explicit_constraint_violations"]) for row in rows),
        "strict_outcomes": dict(sorted(collections.Counter(row["strict_outcome"] for row in rows).items())),
        "target_outcome_categories": sorted(counts),
    }


def evaluate(*, report_paths: dict[str, Path], benchmark_path: Path, tokenizer_path: Path,
             output_json: Path, output_txt: Path) -> dict[str, Any]:
    benchmark_raw = benchmark_path.read_bytes()
    benchmark = json.loads(benchmark_raw)
    tasks = {task["prompt_number"]: task for task in benchmark["tasks"]}
    if len(tasks) != 25:
        raise ValueError("Expected the frozen 25-task executable function benchmark")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    runtime = WasiPython()
    expected_tokenizer_hash = _sha(tokenizer_path)
    benchmark_script_sha = _sha(ROOT / "work/prompt5_dense75_250m_fullgpu_test.py")
    if benchmark.get("benchmark_id") != "simple_python_v1":
        raise ValueError("Unexpected functional benchmark identity")

    all_rows: list[dict[str, Any]] = []
    inputs = {}
    for variant in VARIANTS:
        path = Path(report_paths[variant])
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != 175:
            raise ValueError(f"{variant}: expected 175 benchmark rows, got {len(rows)}")
        summary_path = ROOT / "experiments" / "prompt5_500M_v1" / variant / "summary_500M.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint_path = ROOT / "experiments" / "prompt5_500M_v1" / variant / "checkpoints" / "model_500M.pt"
        full_summary = summary.get("full_state", {})
        full_checkpoint_path = ROOT / full_summary.get("path", "")
        receipt_path = ROOT / "experiments" / "prompt5_500M_v1" / variant / "checkpoint_verification.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        checkpoint_sha = _sha(checkpoint_path)
        full_checkpoint_sha = _sha(full_checkpoint_path)
        final_receipt = receipt.get("final_500M") or {}
        if (summary.get("tokens_seen") != TOKENS_500M or summary.get("model_only_sha256") != checkpoint_sha
                or summary.get("tokenizer_sha256") != expected_tokenizer_hash
                or full_summary.get("tokens_seen") != TOKENS_500M
                or full_summary.get("sha256") != full_checkpoint_sha
                or final_receipt.get("tokens_seen") != TOKENS_500M
                or final_receipt.get("sha256") != full_checkpoint_sha):
            raise ValueError(f"{variant}: summary does not bind the exact 500M model checkpoint")
        expected_keys = {(number, float(temp)) for number in tasks for temp in (0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5)}
        found = set()
        per_variant = []
        for source in rows:
            key = (int(source["prompt_number"]), float(source["temperature"]))
            if key in found or key not in expected_keys:
                raise ValueError(f"{variant}: duplicate or unexpected task-temperature row {key}")
            found.add(key)
            task = tasks[key[0]]
            if (source.get("input") != task["prompt"] or source.get("training_tokens") != TOKENS_500M
                    or source.get("checkpoint_sha256") != checkpoint_sha
                    or source.get("tokenizer_sha256") != expected_tokenizer_hash
                    or source.get("benchmark_source_sha256") != benchmark_script_sha):
                raise ValueError(f"{variant}: output row does not match its frozen prompt/checkpoint/tokenizer")
            scored = _score_row(source, task, runtime, tokenizer)
            per_variant.append(scored)
        if found != expected_keys:
            raise ValueError(f"{variant}: incomplete matched generation matrix")
        all_rows.extend(per_variant)
        inputs[variant] = {"generation_file": path.name, "generation_sha256": hashlib.sha256(payload).hexdigest(),
                           "generation_rows": len(rows), "checkpoint_sha256": checkpoint_sha,
                           "checkpoint_bytes": checkpoint_path.stat().st_size,
                           "full_state_sha256": full_checkpoint_sha,
                           "full_state_bytes": full_checkpoint_path.stat().st_size,
                           "summary_sha256": _sha(summary_path), "verification_sha256": _sha(receipt_path)}

    grouped = {variant: [r for r in all_rows if r["variant"] == variant] for variant in VARIANTS}
    by_temperature = {variant: {f"{temp:.1f}": _summary([r for r in rows if float(r["temperature"]) == temp])
                                for temp in (0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5)}
                      for variant, rows in grouped.items()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    report = {
        "schema_version": 1, "study": "matched 500M executable target-function evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(), "gpu_used_for_scoring": False,
        "benchmark": {"file": benchmark_path.name, "sha256": hashlib.sha256(benchmark_raw).hexdigest(),
                      "benchmark_id": benchmark["benchmark_id"], "tasks": len(tasks),
                      "temperatures": [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5],
                      "generation_script_sha256": benchmark_script_sha},
        "tokenizer_sha256": expected_tokenizer_hash,
        "runtime": runtime.runtime_identity, "inputs": inputs,
        "provenance": {"git_commit": commit, "git_dirty": dirty,
                       "scorer_sha256": _sha(Path(__file__)),
                       "target_extractor_sha256": file_hash(ROOT / "src/eval/target_function.py"),
                       "wasi_runner_sha256": file_hash(ROOT / "src/eval/wasi_python.py")},
        "aggregate": {variant: _summary(rows) for variant, rows in grouped.items()},
        "by_temperature": by_temperature,
        "completions": all_rows,
        "limitations": [
            "Target outcomes execute the earliest syntactically complete requested function with the same hidden cases across variants.",
            "Strict outcomes execute the entire parseable model completion; Markdown fences are normalized by the target extractor before execution.",
            "The loop, empty-input, first-occurrence-order, and fresh-list constraints receive explicit checks when stated by the frozen prompt.",
            "Natural stopping is inferred from generated token count below the cap because the saved rows do not preserve EOS token identity.",
            "Tokens after target are approximate re-encodings of decoded text using the frozen tokenizer.",
            "This small synthetic-function benchmark is evidence for this task set only and does not establish general coding ability."
        ]
    }
    output_json, output_txt = Path(output_json), Path(output_txt)
    for path in (output_json, output_txt):
        if path.exists():
            raise FileExistsError(f"Preserving existing report: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive_json(output_json, report)
    lines = ["Matched 500M executable target-function evaluation", "GPU used for scoring: no", ""]
    for variant in VARIANTS:
        row = report["aggregate"][variant]
        lines.append(f"{variant}: {json.dumps(row, sort_keys=True)}")
    lines += ["", f"Rows scored: {len(all_rows)}", f"Benchmark SHA-256: {report['benchmark']['sha256']}",
              f"Tokenizer SHA-256: {expected_tokenizer_hash}",
              "Target and strict full-output scores are reported separately.",
              "Stopping and tail token counts are inferred/approximate as documented in the JSON report."]
    with output_txt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    return report


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", default="results/prompt5_500M_v1/final_matched_functional")
    args = parser.parse_args()
    base = ROOT / "results/prompt5_500M_v1/milestone_benchmarks"
    reports = {variant: base / f"{variant}_500M_fullgpu_benchmark.jsonl" for variant in VARIANTS}
    stem = ROOT / args.output_stem
    report = evaluate(report_paths=reports, benchmark_path=ROOT / "configs/eval/simple_python_v1.json",
                      tokenizer_path=ROOT / "data/research_v1/tokenizer.json",
                      output_json=stem.with_suffix(".json"), output_txt=stem.with_suffix(".txt"))
    print(json.dumps({"samples": len(report["completions"]), "output_json": str(stem.with_suffix('.json')),
                      "output_txt": str(stem.with_suffix('.txt'))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

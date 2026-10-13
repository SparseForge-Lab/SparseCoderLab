"""Rescore saved Prompt-5 completions by their first complete target function."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer

from src.eval.target_function import score_target
from src.eval.wasi_python import WasiPython, file_hash
from tools.score_saved_python import repetition


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _rate(rows, pred):
    return sum(bool(pred(row)) for row in rows) / len(rows) if rows else None


def _summary(rows: list[dict]) -> dict:
    counts = collections.Counter(row["target_outcome"] for row in rows)
    strict_counts = collections.Counter(row.get("strict_outcome", "missing") for row in rows)
    after = [r["target_tokens_after"] for r in rows if isinstance(r.get("target_tokens_after"), int)]
    return {
        "completions": len(rows), "target_outcomes": dict(sorted(counts.items())),
        "target_parseable_count": sum(bool(r["target_parseable"]) for r in rows),
        "target_pass_at_1": _rate(rows, lambda r: r["target_functional_correct"]),
        "strict_full_completion_outcomes": dict(sorted(strict_counts.items())),
        "natural_stop_inferred_count": sum(r["termination_inferred"] == "stopped_before_cap" for r in rows),
        "cap_reached_count": sum(r["termination_inferred"] == "cap_reached" for r in rows),
        "target_complete_count": sum(r["target_status"] == "complete" for r in rows),
        "target_completed_by_cap_fraction": _rate(rows, lambda r: r["target_status"] == "complete"),
        "target_after_tokens_mean_reencoded": sum(after) / len(after) if after else None,
        "later_code_count": sum(r.get("trailing_nonblank_lines", 0) > 0 for r in rows),
        "syntax_damage_after_target_count": sum(bool(r.get("syntax_damage_after_target")) for r in rows),
        "later_repetition_count": sum(bool(r.get("later_repetition")) for r in rows),
        "multiple_target_definition_count": sum(r.get("target_definitions", 0) > 1 for r in rows),
        "later_redefinition_overwrites_earlier_correct_count": sum(
            bool(r.get("later_redefinition_overwrites_earlier_correct")) for r in rows),
        "strict_pass_but_target_fail_count": sum(r.get("strict_outcome") == "correct" and not r["target_functional_correct"] for r in rows),
        "target_pass_but_strict_fail_count": sum(r["target_functional_correct"] and r.get("strict_outcome") != "correct" for r in rows),
    }


def _pair(rows_by_cap: dict[int, list[dict]]) -> dict:
    left = {(r["variant"], r["prompt_number"]): r for r in rows_by_cap[128]}
    right = {(r["variant"], r["prompt_number"]): r for r in rows_by_cap[256]}
    if left.keys() != right.keys():
        raise ValueError(f"Matched 128/256 rows differ: {len(left)} vs {len(right)}")
    pairs = []
    for key in sorted(left):
        a, b = left[key], right[key]
        ca, cb = a["raw_completion"], b["raw_completion"]
        pairs.append({"variant": key[0], "prompt_number": key[1], "task_id": a["task_id"],
                      "128_completion_is_prefix_of_256": cb.startswith(ca),
                      "target_completed_only_by_256": a["target_status"] != "complete" and b["target_status"] == "complete",
                      "target_pass_changed": a["target_functional_correct"] != b["target_functional_correct"],
                      "target_pass_improved": not a["target_functional_correct"] and b["target_functional_correct"],
                      "target_code_changed": a.get("target_code_sha256") != b.get("target_code_sha256"),
                      "strict_outcome_128": a.get("strict_outcome"), "strict_outcome_256": b.get("strict_outcome"),
                      "strict_pass_lost": a.get("strict_outcome") == "correct" and b.get("strict_outcome") != "correct",
                      "strict_pass_gained": a.get("strict_outcome") != "correct" and b.get("strict_outcome") == "correct",
                      "target_completed_at_128": a["target_status"] == "complete",
                      "same_target_and_scores": (a.get("target_code_sha256") == b.get("target_code_sha256")
                          and a["target_outcome"] == b["target_outcome"]
                          and a.get("strict_outcome") == b.get("strict_outcome")),
                      "later_repetition_at_256": bool(b.get("later_repetition")),
                      "syntax_damage_after_target_at_256": bool(b.get("syntax_damage_after_target")),
                      "later_redefinition_at_256": b.get("target_definitions", 0) > 1})
    counts = {field: sum(bool(row[field]) for row in pairs) for field in pairs[0] if isinstance(pairs[0][field], bool)}
    by_variant = {}
    for variant in sorted({p["variant"] for p in pairs}):
        subset = [p for p in pairs if p["variant"] == variant]
        by_variant[variant] = {field: sum(bool(row[field]) for row in subset)
                               for field in counts}
    return {"pair_count": len(pairs), "counts": counts, "by_variant": by_variant, "pairs": pairs}


def run(args) -> None:
    benchmark_path, tokenizer_path = Path(args.benchmark), Path(args.tokenizer)
    benchmark_raw = benchmark_path.read_bytes()
    benchmark = json.loads(benchmark_raw)
    task_by_prompt = {task["prompt_number"]: task for task in benchmark["tasks"]}
    runtime, tokenizer = WasiPython(), Tokenizer.from_file(str(tokenizer_path))
    caps = {128: (Path(args.generations_128), Path(args.strict_128)),
            256: (Path(args.generations_256), Path(args.strict_256))}
    all_rows, inputs, started = {}, {}, time.perf_counter()
    for cap, (source_path, strict_path) in caps.items():
        source_raw, strict_raw = source_path.read_bytes(), strict_path.read_bytes()
        source_rows = [json.loads(line) for line in source_raw.decode("utf8").splitlines() if line.strip()]
        strict = json.loads(strict_raw)
        source_digest = hashlib.sha256(source_raw).hexdigest()
        if strict.get("source_sha256") != source_digest:
            raise ValueError(f"Strict report at cap {cap} does not bind the selected generation file")
        if strict.get("benchmark_sha256") != hashlib.sha256(benchmark_raw).hexdigest():
            raise ValueError(f"Strict report at cap {cap} does not bind the selected benchmark")
        strict_by_key = {(r["variant"], r["prompt_number"]): r for r in strict["completions"]}
        if len(source_rows) != 400 or len(strict_by_key) != len(source_rows):
            raise ValueError(f"Expected 400 matched source and strict rows at cap {cap}")
        expected_keys = {(variant, number) for variant in {
            "dense75_ref", "sparse75", "sparse75_ngram10m", "sparse75_ngram25m"}
                         for number in task_by_prompt}
        source_keys = {(r.get("variant"), r.get("prompt_number")) for r in source_rows}
        if source_keys != expected_keys or set(strict_by_key) != expected_keys:
            raise ValueError(f"Generation matrix is incomplete or duplicated at cap {cap}")
        rows = []
        for source in source_rows:
            task = task_by_prompt.get(source["prompt_number"])
            if task is None or source.get("input") != task["prompt"]:
                raise ValueError(f"Frozen prompt mismatch for {source.get('task_id')}")
            completion = source.get("raw_output", "")
            result = score_target(completion, task["prompt"], task, runtime, tokenizer=tokenizer)
            end = result.get("target_end_char")
            result.update({
                "cap": cap, "variant": source["variant"], "task_id": task["id"],
                "family_id": task.get("family_id", task["id"]), "category": task.get("category"),
                "prompt_number": task["prompt_number"], "prompt": task["prompt"],
                "raw_completion": completion,
                "completion_sha256": hashlib.sha256(completion.encode("utf8")).hexdigest(),
                "source_generated_tokens": source.get("generated_tokens"),
                "source_max_new_tokens": source.get("max_new_tokens"),
                "termination_inferred": ("cap_reached" if source.get("generated_tokens", 0) >= cap
                                         else "stopped_before_cap"),
                "target_tokens_before": (len(tokenizer.encode(completion[:end]).ids) if end is not None else None),
                "later_repetition": repetition(completion[end:]) if end is not None else False,
                "strict_outcome": strict_by_key[(source["variant"], source["prompt_number"])].get("outcome"),
                "strict_parseable": strict_by_key[(source["variant"], source["prompt_number"])].get("parseable"),
                "strict_functional_correct": strict_by_key[(source["variant"], source["prompt_number"])].get("functional_correct"),
                "truncated": strict_by_key[(source["variant"], source["prompt_number"])].get("truncated"),
                "checkpoint": source.get("checkpoint"), "checkpoint_sha256": source.get("checkpoint_sha256"),
                "checkpoint_manifest_sha256": source.get("checkpoint_manifest_sha256"),
                "training_tokens": source.get("training_tokens"),
                "tokenizer_sha256": source.get("tokenizer_sha256"),
                "benchmark_sha256": source.get("benchmark_sha256"),
                "generation_git_commit": source.get("generation_git_commit"),
                "generation_git_dirty": source.get("generation_git_dirty"),
            })
            result["target_code_sha256"] = (hashlib.sha256(result["target_code"].encode()).hexdigest()
                                              if result.get("target_code") else None)
            rows.append(result)
        all_rows[cap] = rows
        inputs[str(cap)] = {"generation_file": source_path.name, "generation_sha256": source_digest,
                            "generation_bytes": len(source_raw), "strict_report": strict_path.name,
                            "strict_report_sha256": hashlib.sha256(strict_raw).hexdigest(),
                            "generation_row_count": len(source_rows)}
    # Confirm input immutability after the WASI pass.
    for cap, (source_path, strict_path) in caps.items():
        if sha256(source_path) != inputs[str(cap)]["generation_sha256"] or sha256(strict_path) != inputs[str(cap)]["strict_report_sha256"]:
            raise ValueError("Saved generation or strict report changed while scoring")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    report = {
        "schema_version": 1, "study": "Prompt-5 matched 250M target-function audit",
        "created_utc": datetime.now(timezone.utc).isoformat(), "gpu_used": False,
        "target_scoring_method": "First complete requested top-level function extracted from parseable AST line prefixes; no syntax repair. Only referenced prior imports, function helpers, or literal assignments are included. Extracted code is executed in pinned CPython/WASI.",
        "token_count_method": "Tokenizer re-encodes decoded text before/after target end; counts are approximate because original generated token IDs and character offsets were not saved.",
        "benchmark": {"file": benchmark_path.name, "sha256": hashlib.sha256(benchmark_raw).hexdigest(),
                      "tokenizer_file": tokenizer_path.name, "tokenizer_sha256": sha256(tokenizer_path),
                      "source_benchmark_id": benchmark.get("benchmark_id")},
        "inputs": inputs,
        "provenance": {"git_commit": commit, "git_dirty": dirty,
                       "target_scorer_sha256": file_hash(__file__),
                       "target_extractor_sha256": file_hash("src/eval/target_function.py"),
                       "wasi_runner_sha256": file_hash("src/eval/wasi_python.py"),
                       "runtime": runtime.runtime_identity,
                       "python": sys.version, "platform": platform.platform(),
                       "scoring_seconds": time.perf_counter() - started},
        "caps": {str(cap): {"summary": _summary(rows), "completions": rows}
                 for cap, rows in all_rows.items()},
        "matched_128_to_256": _pair(all_rows),
        "limitations": [
            "The target-function score answers whether the first complete requested function passes fixed cases; it is distinct from the original strict whole-completion score, which is retained per row.",
            "Target completion positions and tokens after the target are approximate re-encodings of decoded text, not original generated-token offsets.",
            "A complete AST prefix can still be a semantically incomplete implementation; fixed benchmark cases determine functional pass.",
            "Only referenced, safely identified prior imports, function helpers, and literal assignments are retained. Required dependencies outside that set are classified as extraction failure or runtime failure.",
            "Natural stopping is inferred from generated token count being below the configured cap; source rows do not preserve EOS token identity.",
        ],
    }
    output = Path(args.output)
    if output.exists() or output.with_suffix(".txt").exists():
        raise FileExistsError("Choose fresh report paths; existing results are preserved")
    dump_new(output, report)
    text_path = output.with_suffix(".txt")
    lines = ["Prompt-5 matched 250M target-function audit", "GPU used: no", ""]
    for cap, value in report["caps"].items():
        lines.append(f"Cap {cap}: {json.dumps(value['summary'], sort_keys=True)}")
        for variant in sorted({r["variant"] for r in value["completions"]}):
            lines.append(f"Cap {cap} / {variant}: {json.dumps(_summary([r for r in value['completions'] if r['variant'] == variant]), sort_keys=True)}")
    lines += ["", f"128→256 matched comparison: {json.dumps(report['matched_128_to_256']['counts'], sort_keys=True)}",
              "", "Strict full-completion outcomes are included alongside the separate target-function score.",
              "Target locations and tail token counts are approximate re-encodings of decoded text."]
    with text_path.open("x", encoding="utf8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"json": str(output), "text": str(text_path), "gpu_used": False,
                      "rows": {str(cap): len(rows) for cap, rows in all_rows.items()},
                      "scoring_seconds": report["provenance"]["scoring_seconds"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="configs/eval/unified_python_v1_prompt5_128baseline.json")
    parser.add_argument("--tokenizer", default="data/research_v1/tokenizer.json")
    parser.add_argument("--generations-128", default="results/unified_python_250M_greedy_r1.jsonl")
    parser.add_argument("--strict-128", default="results/unified_python_250M_functional_128tok_truncation_scored_r2.json")
    parser.add_argument("--generations-256", default="results/unified_python_250M_greedy_256tok_prompt5baseline_r1.jsonl")
    parser.add_argument("--strict-256", default="results/unified_python_250M_functional_256tok_truncation_scored_r1.json")
    parser.add_argument("--output", default="results/unified_python_250M_target_function_analysis_r1.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

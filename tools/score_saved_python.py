"""Score frozen, matched completions using a CPU-only CPython/WASI guest."""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import re
from pathlib import Path

from src.eval.wasi_python import WasiPython, file_hash


def equivalent(actual, expected):
    if isinstance(expected, bool) or expected is None:
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)):
        if type(actual) not in (int, float):
            return False
        try:
            return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        except OverflowError:
            return False
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            equivalent(a, e) for a, e in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def repetition(text):
    """Same adjacent token-repeat heuristic used by the saved generation study."""
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) >= 12:
        for size in range(3, min(12, len(tokens) // 2 + 1)):
            if any(tokens[i:i+size] == tokens[i+size:i+2*size] for i in range(len(tokens)-2*size+1)):
                return True
    return bool(re.search(r"(.{8,}?)(?:\1){2,}", text, re.S))


def score(runtime, source, task):
    repeat = repetition(source[len(task["prompt"]):])
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {"outcome": "syntax_error", "parseable": False, "repetition": repeat,
                "functional_correct": False, "passed_cases": 0, "case_count": len(task["cases"])}
    execution = runtime.execute(source, task["function"], [c["args"] for c in task["cases"]])
    outcome = execution["kind"]
    passed = 0
    if outcome == "completed":
        observations = execution.get("results")
        if not isinstance(observations, list) or len(observations) != len(task["cases"]):
            outcome = "invalid_output"
        else:
            passed = sum(isinstance(o, dict) and set(o) == {"value", "stdout"} and equivalent(o["value"], c["expected"])
                         and o.get("stdout") == c["stdout"] for o, c in zip(observations, task["cases"]))
            outcome = "correct" if passed == len(task["cases"]) else "wrong"
    has_loop = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    return {"outcome": outcome, "parseable": True, "repetition": repeat,
            "functional_correct": outcome == "correct", "passed_cases": passed,
            "case_count": len(task["cases"]), "uses_loop": has_loop,
            "loop_constraint_met": task["id"] != "sum_values" or has_loop,
            "execution_seconds": execution["wall_seconds"], "fuel_consumed": execution["fuel_consumed"]}


def validate_rows(rows, tasks, checkpoint_manifest):
    variants = set(checkpoint_manifest["canonical_transition_checkpoints"])
    lookup = {t["prompt_number"]: t for t in tasks}
    temperatures = {0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5}
    seen = set()
    policies = collections.defaultdict(set)
    hash_attested = any('checkpoint_sha256' in item for item in rows)
    for row in rows:
        variant = row["variant"]
        if variant not in variants or row["temperature"] not in temperatures:
            raise ValueError("Unexpected model or temperature")
        task = lookup[row["prompt_number"]]
        if task["prompt"] != row["input"]:
            raise ValueError("Prompt mismatch")
        canonical = checkpoint_manifest["canonical_transition_checkpoints"][variant]
        if hash_attested:
            if row.get('checkpoint_sha256') != canonical['model_weights']['sha256']:
                raise ValueError('Generation checkpoint hash mismatch')
        if row["training_tokens"] != canonical["canonical_transition_tokens"]:
            raise ValueError("Unmatched checkpoint token stage")
        # Original path is private metadata; only check its variant membership.
        normalized = row["checkpoint"].replace("\\", "/")
        if f"/{variant}/checkpoints/" not in normalized:
            raise ValueError("Checkpoint model/path mismatch")
        key = (variant, row["temperature"], row["prompt_number"])
        if key in seen:
            raise ValueError("Duplicate completion slot")
        seen.add(key)
        policies[row["temperature"]].add(json.dumps({k: row[k] for k in (
            "generation_seed", "max_new_tokens", "top_p", "top_k", "repetition_penalty", "stop_tokens", "device")}, sort_keys=True))
    if seen != {(v, temp, t) for v in variants for temp in temperatures for t in lookup} or any(len(p) != 1 for p in policies.values()):
        raise ValueError("Missing/unmatched model prompt or decoding condition")
    return {str(t): json.loads(next(iter(p))) for t,p in sorted(policies.items())}


def summarize(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["temperature"])].append(row)
    summaries = []
    for (variant, temperature), group in sorted(groups.items()):
        count = len(group)
        summaries.append({"variant": variant, "temperature": temperature, "tasks": count,
                          "correct": sum(r["functional_correct"] for r in group),
                          "functional_rate": sum(r["functional_correct"] for r in group) / count,
                          "parseable_rate": sum(r["parseable"] for r in group) / count,
                          "repetition_rate": sum(r["repetition"] for r in group) / count,
                          "correct_without_repetition": sum(r["functional_correct"] and not r["repetition"] for r in group),
                          "outcomes": dict(collections.Counter(r["outcome"] for r in group))})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/prompt3_cpu_generation_benchmark.jsonl")
    parser.add_argument("--output", default="results/research_v2_real/functional_100M_v1.json")
    parser.add_argument("--benchmark", default="configs/eval/simple_python_v1.json")
    parser.add_argument("--manifest", default="results/prompt3/transition_checkpoint_manifest.json")
    args = parser.parse_args()
    raw = Path(args.input).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf8").splitlines() if line.strip()]
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf8"))
    policy = validate_rows(rows, benchmark["tasks"], manifest)
    lookup = {t["prompt_number"]: t for t in benchmark["tasks"]}
    runtime = WasiPython()
    results = []
    for i, row in enumerate(rows):
        task = lookup[row["prompt_number"]]
        source = row["input"] + row["raw_output"]
        result = score(runtime, source, task)
        result.update(variant=row["variant"], training_tokens=row["training_tokens"],
                      temperature=row["temperature"], task_id=task["id"],
                      completion_sha256=hashlib.sha256(row["raw_output"].encode()).hexdigest())
        results.append(result)
        if (i+1) % 175 == 0:
            print(json.dumps({"scored": i+1, "total": len(rows), "gpu_used": False}), flush=True)
    # Refuse a moving source; never silently combine generations from active jobs.
    if Path(args.input).read_bytes() != raw:
        raise ValueError("Generation file changed during scoring")
    report = {"schema_version": 1, "benchmark_id": benchmark["benchmark_id"],
              "benchmark_sha256": file_hash(args.benchmark), "source_file": Path(args.input).name,
              "source_sha256": hashlib.sha256(raw).hexdigest(), "source_bytes": len(raw),
              "manifest_sha256": file_hash(args.manifest), "decoding_policy": policy,
              "runtime": runtime.runtime_identity, "scope": benchmark["scope"],
              "scorer_sha256": file_hash(__file__),
              "sandbox_source_sha256": file_hash("src/eval/wasi_python.py"),
              "generation_checkpoint_hashes_verified": all('checkpoint_sha256' in row for row in rows),
              "limitations": [("Generation records carry canonical checkpoint hashes checked by the sampler before/after loading; this is local provenance evidence, not a cryptographic execution attestation." if all('checkpoint_sha256' in row for row in rows) else "Saved generations declare model identity and token counts but contain no checkpoint SHA256 at generation time; canonical manifest hashes cannot retroactively prove weight identity."),
                              "Small finite test suite; not pass@k, instruction-following, repository engineering, or benchmark leadership.",
                              "Repetition is a heuristic and is reported independently of functional correctness.",
                              "Parseability is the host Python AST check; executed candidates use CPython 3.14.7 in WASI.",
                              "Guest linear-memory cap is not a bound on the host's total RSS. Compilation time is outside each guest execution deadline."],
              "canonical_checkpoint_references": {v: m["model_weights"] for v,m in manifest["canonical_transition_checkpoints"].items()},
              "summaries": summarize(results), "completions": results}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Choose a new report version; preserve existing results")
    temporary = destination.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    temporary.replace(destination)
    print(json.dumps({"completions": len(results), "output": args.output, "gpu_used": False}), flush=True)


if __name__ == "__main__":
    main()

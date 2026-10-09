"""Unified CPU-only functional scoring, telemetry and planning CLI.

Scoring consumes saved generations. Discovery only writes a manual queue; no
mode in this module loads a checkpoint or starts CUDA/model work.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
import sys
import time
import textwrap
from pathlib import Path

from src.eval.unified import (architecture_estimate, checkpoint_candidates,
                              efficiency_summary, ngram_summary, routing_summary,
                              classify_completion, summarize_completions)
from src.eval.wasi_python import WasiPython, file_hash


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def write_new(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_generation_matrix(generations: list[dict], tasks: list[dict], expected_variants=()) -> set:
    seen = set()
    for row in generations:
        key = (row.get("variant", "unknown"), row.get("temperature"), row.get("prompt_number"))
        if key in seen:
            raise ValueError(f"Duplicate generation slot: {key}")
        seen.add(key)
    variants = set(expected_variants)
    if variants:
        expected = {(variant, 0.0, task["prompt_number"]) for variant in variants for task in tasks}
        if seen != expected:
            raise ValueError(f"Matched benchmark matrix incomplete or unexpected (missing={len(expected-seen)}, unexpected={len(seen-expected)})")
    return seen


def validate_policy_consistency(generations: list[dict], *, canonical_greedy: bool = False) -> dict:
    keys = ("decoding", "deterministic_algorithms", "gpu_name", "generation_seed", "max_new_tokens", "top_p", "top_k",
            "repetition_penalty", "stop_tokens", "device", "bf16", "tf32", "generation_batch_size")
    policies = {json.dumps({key: row.get(key) for key in keys}, sort_keys=True) for row in generations}
    if len(policies) > 1:
        raise ValueError("Generation decoding/precision policy differs between rows")
    policy = json.loads(next(iter(policies))) if policies else {}
    if canonical_greedy and (policy.get("decoding") != "greedy"
                            or policy.get("gpu_name") != "NVIDIA GeForce RTX 5070"
                            or policy.get("generation_seed") != 42 or policy.get("max_new_tokens") != 128
                            or policy.get("deterministic_algorithms") is not True or policy.get("device") != "cuda"
                            or policy.get("bf16") is not True or policy.get("tf32") is not True
                            or policy.get("top_p") is not None or policy.get("top_k") is not None
                            or policy.get("repetition_penalty") != 1.0 or policy.get("generation_batch_size") != 4):
        raise ValueError("Expected the frozen deterministic greedy generation policy")
    return policy


def deduplicate_candidates(candidates: list[dict], previously_seen: set[str]) -> list[dict]:
    result = []
    seen = set(previously_seen)
    for row in candidates:
        digest = row["sha256"]
        if digest not in seen:
            result.append(row)
            seen.add(digest)
    return result


def structural_requirements(task: dict, prompt: str, extracted_code: str | None) -> dict:
    if task.get("family_id", task.get("id", "")).startswith("sum_values"):
        code = extracted_code or ""
        has_definition = bool(re.search(r"(?m)^\s*def\s+sum_values\s*\(", code))
        source = code if has_definition else prompt + "\n" + textwrap.indent(code, "    ")
        try:
            tree = ast.parse(source)
            uses_loop = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
        except (SyntaxError, ValueError):
            uses_loop = False
        return {"loop_constraint_required": True, "uses_loop": uses_loop,
                "loop_constraint_met": uses_loop}
    return {"loop_constraint_required": False, "uses_loop": None, "loop_constraint_met": None}


def summarize_group(rows: list[dict]) -> dict:
    summary = summarize_completions(rows)
    constrained = [row for row in rows if row.get("loop_constraint_required")]
    summary["loop_constraint_rate"] = (sum(bool(row.get("loop_constraint_met")) for row in constrained) / len(constrained)
                                       if constrained else None)
    summary["loop_constraint_tasks"] = len(constrained)
    return summary


def score(args) -> None:
    benchmark_path, source_path, output_path = map(Path, (args.benchmark, args.generations, args.output))
    if output_path.exists() or output_path.with_suffix(".txt").exists():
        raise FileExistsError("Choose fresh JSON and text report paths; existing results are preserved")
    raw = source_path.read_bytes()
    benchmark_raw = benchmark_path.read_bytes()
    efficiency_path = Path(args.efficiency) if args.efficiency else None
    efficiency_raw = efficiency_path.read_bytes() if efficiency_path else None
    benchmark = json.loads(benchmark_raw)
    tasks = benchmark.get("tasks", [])
    if not 100 <= len(tasks) <= 300:
        raise ValueError(f"Unified benchmark must contain 100–300 frozen tasks; found {len(tasks)}")
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Task IDs must be unique")
    runtime = WasiPython()
    task_by_prompt = {task["prompt_number"]: task for task in tasks}
    generations = [json.loads(line) for line in raw.decode("utf8").splitlines() if line.strip()]
    validate_generation_matrix(generations, tasks, args.expected_variants or [])
    generation_policy = validate_policy_consistency(generations, canonical_greedy=bool(args.expected_variants))
    if args.expected_variants:
        expected_benchmark_sha = hashlib.sha256(benchmark_raw).hexdigest()
        if any(row.get("benchmark_sha256") != expected_benchmark_sha for row in generations):
            raise ValueError("Generation rows do not bind this frozen benchmark")
        tokenizer_hashes = {row.get("tokenizer_sha256") for row in generations}
        if len(tokenizer_hashes) != 1 or not isinstance(next(iter(tokenizer_hashes), None), str) or len(next(iter(tokenizer_hashes))) != 64:
            raise ValueError("Matched generation rows need one tokenizer SHA-256")
    checkpoint_hashes = {}
    results = []
    start = time.perf_counter()
    for row in generations:
        if row.get("prompt_number") not in task_by_prompt:
            raise ValueError(f"Unknown task prompt_number: {row.get('prompt_number')}")
        task = task_by_prompt[row["prompt_number"]]
        if row.get("input", task["prompt"]) != task["prompt"]:
            raise ValueError(f"Frozen prompt mismatch for {task['id']}")
        if args.expected_training_tokens is not None and row.get("training_tokens") != args.expected_training_tokens:
            raise ValueError(f"Unexpected checkpoint token stage for {row.get('variant')}")
        checkpoint_hash = row.get("checkpoint_sha256")
        if args.expected_variants and (not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64):
            raise ValueError("Matched benchmark rows must carry checkpoint SHA-256")
        if checkpoint_hash:
            previous_hash = checkpoint_hashes.setdefault(row.get("variant", "unknown"), checkpoint_hash)
            if previous_hash != checkpoint_hash:
                raise ValueError(f"Checkpoint hash changed within variant {row.get('variant')}")
        completion = row.get("raw_output", "")
        result = classify_completion(completion, row.get("input", task["prompt"]), task, runtime,
                                     token_limit=row.get("max_new_tokens"),
                                     generated_tokens=row.get("generated_tokens"))
        result.update(task_id=task["id"], variant=row.get("variant", "unknown"),
                      temperature=row.get("temperature"), prompt_number=row["prompt_number"],
                      family_id=task.get("family_id", task["id"]), category=task.get("category", "uncategorized"),
                      paraphrase_index=task.get("paraphrase_index"),
                      generation_policy={key: row.get(key) for key in ("decoding", "deterministic_algorithms", "gpu_name", "generation_seed", "max_new_tokens", "top_p", "top_k", "repetition_penalty", "stop_tokens", "device", "bf16", "tf32", "generation_batch_size")},
                      prompt=row.get("input", task["prompt"]), raw_completion=completion,
                      completion_sha256=hashlib.sha256(completion.encode("utf8")).hexdigest(),
                      checkpoint=row.get("checkpoint"), checkpoint_sha256=row.get("checkpoint_sha256"),
                      training_tokens=row.get("training_tokens"), generated_tokens=row.get("generated_tokens"),
                      generation_seconds=row.get("generation_group_seconds"),
                      tokens_per_second=(row["generated_tokens"] / row["generation_group_seconds"]
                                         if row.get("generated_tokens") is not None and row.get("generation_group_seconds") else None),
                      model_load_seconds=row.get("model_load_seconds"),
                      peak_vram_bytes=row.get("peak_allocated_bytes"),
                      checkpoint_bytes=row.get("checkpoint_bytes"))
        result.update(structural_requirements(task, row.get("input", task["prompt"]), result.get("extracted_code")))
        results.append(result)
    elapsed = time.perf_counter() - start
    if (source_path.read_bytes() != raw or benchmark_path.read_bytes() != benchmark_raw
            or (efficiency_path and efficiency_path.read_bytes() != efficiency_raw)):
        raise ValueError("Frozen input changed while scoring")
    groups = {}
    keys = sorted({(r["variant"], r["temperature"]) for r in results}, key=lambda x: (x[0], str(x[1])))
    for variant, temperature in keys:
        group = [r for r in results if r["variant"] == variant and r["temperature"] == temperature]
        groups[f"{variant}/{temperature}"] = summarize_group(group)
        for category in sorted({r["category"] for r in group}):
            subgroup = [r for r in group if r["category"] == category]
            groups[f"{variant}/{temperature}/category:{category}"] = summarize_group(subgroup)
    comparisons = {}
    for temperature in sorted({r["temperature"] for r in results}, key=str):
        categories = [None] + sorted({r["category"] for r in results})
        for category in categories:
            comparison_key = f"temperature:{temperature}/category:{category or 'all'}"
            comparisons[comparison_key] = {}
            for variant in sorted({r["variant"] for r in results}):
                subset = [r for r in results if r["variant"] == variant and r["temperature"] == temperature
                          and (category is None or r["category"] == category)]
                if subset:
                    summary = summarize_group(subset)
                    comparisons[comparison_key][variant] = {"completions": summary["completions"],
                                                             "functional_accuracy": summary["functional_accuracy"],
                                                             "parseable_rate": summary["parseable_rate"],
                                                             "repetition_rate": summary["repetition_rate"],
                                                             "loop_constraint_rate": summary["loop_constraint_rate"],
                                                             "outcomes": summary["outcomes"]}
    efficiency_rows = [json.loads(line) for line in efficiency_raw.decode("utf8").splitlines() if line.strip()] if efficiency_raw is not None else []
    efficiency_by_variant = {variant: efficiency_summary([row for row in efficiency_rows if row.get("variant") == variant])
                             for variant in sorted({row.get("variant", "unknown") for row in efficiency_rows})}
    report = {"schema_version": 1, "benchmark_id": benchmark.get("benchmark_id"),
              "benchmark_sha256": hashlib.sha256(benchmark_raw).hexdigest(),
              "source_file": source_path.name, "source_sha256": hashlib.sha256(raw).hexdigest(),
              "source_bytes": len(raw), "generation_policy": generation_policy, "runtime": runtime.runtime_identity,
              "scorer_sha256": file_hash(__file__), "aggregation_sha256": file_hash("src/eval/unified.py"),
              "task_count": len(tasks), "completion_count": len(results), "groups": groups,
              "matched_comparisons": comparisons,
              "efficiency": efficiency_summary(efficiency_rows or results),
              "efficiency_by_variant": efficiency_by_variant,
              "efficiency_source": ({"file": efficiency_path.name, "sha256": hashlib.sha256(efficiency_raw).hexdigest()} if efficiency_path else None),
              "scoring_seconds": elapsed, "gpu_used": False,
              "environment": {"python": sys.version, "platform": platform.platform()},
              "limitations": ["Parseability and functional correctness are distinct; parseable code can be wrong.",
                              "The task suite has four correlated wordings per 25 underlying functions and does not measure agentic repository work.",
                              "WASI guest limits do not cap host total RSS; see runtime identity and evaluator documentation.",
                              "Generation throughput and model-load measurements are copied only when present in source records; absent values remain null.",
                              "Greedy throughput includes the first measured prompt batch; it is not a warmed steady-state kernel benchmark."],
              "completions": results}
    write_new(output_path, report)
    text_path = output_path.with_suffix(".txt")
    lines = [f"Unified functional evaluation: {report['benchmark_id']}",
             f"Tasks: {len(tasks)} | completions: {len(results)} | scoring seconds: {elapsed:.3f} | GPU used: no", ""]
    lines += [f"{name}: accuracy={summary['functional_accuracy']} parseable={summary['parseable_rate']} "
              f"repetition={summary['repetition_rate']} loop_constraint_rate={summary['loop_constraint_rate']} "
              f"pass@1={summary['pass_at_1']['estimate']} "
              f"outcomes={summary['outcomes']}" for name, summary in groups.items()]
    for variant, measurements in efficiency_by_variant.items():
        load = measurements["model_load_seconds"]["sum"]
        gen = measurements["generation_seconds"]["sum"]
        tps = measurements["tokens_per_second"]["sum"]
        vram = measurements["peak_vram_bytes"]["max"]
        rss = measurements["cpu_rss_bytes"]["max"]
        size = measurements["checkpoint_bytes"]["max"]
        lines.append(f"Efficiency {variant}: model_load_s={load} generation_s={gen} generated_tokens_per_s={tps} peak_vram_bytes={vram} peak_cpu_rss_bytes={rss} checkpoint_bytes={size}")
    lines += ["", "Parseability is not correctness. Four prompt wordings per underlying function are correlated. This finite fixed-case suite is not a claim of repository engineering or model leadership."]
    with text_path.open("x", encoding="utf8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"report": str(output_path), "text": str(text_path), "completions": len(results), "gpu_used": False}))


def telemetry(args) -> None:
    records = read_jsonl(Path(args.routes)) if args.routes else []
    ngram_path = Path(args.ngram) if args.ngram else None
    evaluation_path = Path(args.evaluation) if args.evaluation else None
    evaluation = json.loads(evaluation_path.read_text(encoding="utf8")) if evaluation_path else None
    report = {"routing": routing_summary(records) if records else {"available": False, "reason": "No route trace supplied; unavailable is not zero."},
              "ngrams": ngram_summary(json.loads(ngram_path.read_text(encoding="utf8")) if ngram_path else None),
              "existing_gate_router": evaluation.get("router") if evaluation else None,
              "existing_gate_ngram": evaluation.get("ngram") if evaluation else None,
              "provenance": {"tool_sha256": file_hash(__file__), "created_utc": datetime.now(timezone.utc).isoformat(),
                             "routes_sha256": file_hash(args.routes) if args.routes else None,
                             "ngram_sha256": file_hash(ngram_path) if ngram_path else None,
                             "evaluation_sha256": file_hash(evaluation_path) if evaluation_path else None,
                             "python": sys.version, "platform": platform.platform()}, "gpu_used": False}
    write_new(Path(args.output), report)
    print(json.dumps({"output": args.output, "gpu_used": False}))


def queue(args) -> None:
    rows = checkpoint_candidates(Path(args.root))
    existing = []
    if args.existing and Path(args.existing).exists():
        prior = json.loads(Path(args.existing).read_text(encoding="utf8"))
        existing = prior.get("new_candidates", []) if isinstance(prior, dict) else prior
    hashes = {row.get("sha256") for row in existing if row.get("sha256")}
    new = deduplicate_candidates(rows, hashes)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Queue output exists; use --existing for deduplication and a fresh --output")
    write_new(output, {"schema_version": 1, "mode": "queue_only_manual_execution", "gpu_used": False,
                       "new_candidates": new, "already_seen_sha256": sorted(hashes),
                       "provenance": {"tool_sha256": file_hash(__file__), "created_utc": datetime.now(timezone.utc).isoformat(),
                                      "scan_root": str(Path(args.root).resolve()), "python": sys.version,
                                      "platform": platform.platform()},
                       "notice": "Discovery never loads models or automatically evaluates checkpoints."})
    print(json.dumps({"queued": len(new), "gpu_used": False, "output": str(output)}))


def import_history(args) -> None:
    source = Path(args.input); raw = source.read_bytes(); old = json.loads(raw)
    normalized = {"schema_version": 1, "imported_from": source.name,
                  "source_sha256": hashlib.sha256(raw).hexdigest(), "historical_prompt": args.prompt,
                  "match_classification": args.match, "match_basis": args.basis,
                  "continuous_learning_claim": False,
                  "scope_note": "Historical import only; Prompt-3 to Prompt-5 continuity is not inferred.",
                  "original_report": old}
    write_new(Path(args.output), normalized)
    print(json.dumps({"output": args.output, "source_sha256": normalized["source_sha256"], "gpu_used": False}))


def architecture(args) -> None:
    result = architecture_estimate(**{k.replace("-", "_"): getattr(args, k.replace("-", "_"))
                                      for k in ("vocab", "d-model", "layers", "dense-ffn", "experts", "top-k", "moe-layers", "ngram-rows", "ngram-banks", "ngram-dim", "dtype-bytes", "optimizer-multiplier")})
    report = {"schema_version": 1, "estimate": result,
              "provenance": {"tool_sha256": file_hash(__file__), "created_utc": datetime.now(timezone.utc).isoformat(),
                             "python": sys.version, "platform": platform.platform()}, "gpu_used": False}
    print(json.dumps(report, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score"); p.add_argument("--benchmark", required=True); p.add_argument("--generations", required=True); p.add_argument("--efficiency"); p.add_argument("--expected-variants", nargs="+"); p.add_argument("--expected-training-tokens", type=int); p.add_argument("--output", required=True); p.set_defaults(func=score)
    p = sub.add_parser("telemetry"); p.add_argument("--routes"); p.add_argument("--ngram"); p.add_argument("--evaluation"); p.add_argument("--output", required=True); p.set_defaults(func=telemetry)
    p = sub.add_parser("queue"); p.add_argument("--root", required=True); p.add_argument("--existing"); p.add_argument("--output", required=True); p.set_defaults(func=queue)
    p = sub.add_parser("import-history"); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--prompt", required=True); p.add_argument("--match", choices=["exact", "near_match", "unmatched"], required=True); p.add_argument("--basis", required=True); p.set_defaults(func=import_history)
    p = sub.add_parser("architecture");
    for name in ("vocab", "d-model", "layers", "dense-ffn"):
        p.add_argument("--" + name, type=int, required=True)
    for name, default in (("experts", 0), ("top-k", 0), ("moe-layers", 0), ("ngram-rows", 0), ("ngram-banks", 0), ("ngram-dim", 0), ("dtype-bytes", 2)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--optimizer-multiplier", type=float, default=2.0)
    p.set_defaults(func=architecture)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()

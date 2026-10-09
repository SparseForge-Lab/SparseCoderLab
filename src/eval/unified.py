"""CPU-only aggregation helpers for reproducible coding evaluations.

This module deliberately does not load a model or decide what checkpoints to
evaluate. It only summarizes frozen completions and telemetry already collected
by the existing generation/evaluation paths.
"""
from __future__ import annotations

import ast
import collections
import contextlib
import hashlib
import math
import re
import textwrap
from pathlib import Path
from typing import Iterable


def extract_code(text: str) -> tuple[str | None, str]:
    """Return the first fenced code block, or a plausible raw code response."""
    blocks = re.findall(r"```(?:[A-Za-z0-9_+.-]+)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return blocks[0].strip("\r\n"), "fenced"
    if re.search(r"(?m)^\s*(?:def |class |from |import |return\b|print\(|for\b|if\b|while\b|[A-Za-z_]\w*\s*=)", text):
        return text.strip("\r\n"), "raw"
    return None, "no_code"


def classify_completion(text: str, prompt: str, task: dict, runtime,
                        *, token_limit: int | None = None, generated_tokens: int | None = None) -> dict:
    """Execute one untrusted candidate using the pinned WASI runtime."""
    from tools.score_saved_python import repetition, equivalent

    code, extraction = extract_code(text)
    repeated = repetition(text)
    row = {"extraction": extraction, "repetition": repeated, "parseable": False,
           "extracted_code": code, "functional_correct": False,
           "passed_cases": 0, "case_count": len(task["cases"])}
    if not text.strip():
        row["outcome"] = "empty"
        return row
    if token_limit is not None and generated_tokens is not None and generated_tokens >= token_limit:
        row["truncated"] = True
    else:
        row["truncated"] = bool(re.search(r"(?i)(?:<truncated>|\[truncated\])\s*$", text))
    if row["truncated"]:
        row["outcome"] = "truncated"
        return row
    if code is None:
        row["outcome"] = "extraction_failure"
        return row
    # Existing generations often contain the requested function header in the
    # prompt. Preserve this standard convention when a completion starts with a
    # body; extraction is otherwise reported independently.
    stripped = code.lstrip()
    complete_definition = re.search(rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(task['function'])}\s*\(", code)
    if complete_definition or stripped.startswith(("import ", "from ")):
        candidate = code
    else:
        body = code if code[:1].isspace() else textwrap.indent(code, "    ")
        candidate = prompt + "\n" + body
    try:
        ast.parse(candidate)
    except (SyntaxError, ValueError):
        row["outcome"] = "syntax_error"
        return row
    row["parseable"] = True
    execution = runtime.execute(candidate, task["function"], [case["args"] for case in task["cases"]])
    kind = execution["kind"]
    if kind == "completed":
        observations = execution.get("results", [])
        passed = sum(isinstance(o, dict) and equivalent(o.get("value"), case["expected"])
                     and o.get("stdout") == case.get("stdout", "")
                     for o, case in zip(observations, task["cases"]))
        row["passed_cases"] = passed
        row["observations"] = observations
        row["execution_stdout"] = execution.get("module_stdout", "")
        row["execution_stderr"] = execution.get("module_stderr", "")
        row["outcome"] = "correct" if len(observations) == len(task["cases"]) and passed == len(task["cases"]) else "wrong"
    elif kind == "timeout":
        row["outcome"] = "timeout"
    elif kind == "syntax_error":
        row["outcome"] = "syntax_error"
    else:
        row["outcome"] = "runtime_error"
    row["functional_correct"] = row["outcome"] == "correct"
    row["execution_seconds"] = execution.get("wall_seconds")
    row["fuel_consumed"] = execution.get("fuel_consumed")
    row["execution_stdout"] = execution.get("module_stdout", "")
    row["execution_stderr"] = execution.get("module_stderr", "") + execution.get("stderr", "")
    row["partial_stdout"] = execution.get("partial_stdout", "")
    row["partial_stderr"] = execution.get("partial_stderr", "")
    if execution.get("exception"):
        row["exception"] = execution["exception"]
    return row


def pass_at_k(results: Iterable[dict], k: int = 1) -> dict:
    """Unbiased pass@k estimator, grouped by task; missing draws are not zero."""
    if k < 1:
        raise ValueError("k must be positive")
    groups = collections.defaultdict(list)
    for row in results:
        groups[row["task_id"]].append(bool(row.get("functional_correct")))
    estimates = []
    excluded = 0
    for values in groups.values():
        n, c = len(values), sum(values)
        if n < k:
            excluded += 1
            continue
        estimates.append(1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k))
    return {"k": k, "estimate": sum(estimates) / len(estimates) if estimates else None,
            "tasks_scored": len(estimates), "tasks_with_too_few_samples": excluded,
            "scope": "Unbiased finite-sample pass@k estimator over observed draws; missing samples are excluded."}


def summarize_completions(results: list[dict]) -> dict:
    counts = collections.Counter(row.get("outcome", "unscored") for row in results)
    total = len(results)
    return {"completions": total, "outcomes": dict(sorted(counts.items())),
            "functional_accuracy": counts["correct"] / total if total else None,
            "parseable_rate": sum(bool(r.get("parseable")) for r in results) / total if total else None,
            "repetition_rate": sum(bool(r.get("repetition")) for r in results) / total if total else None,
            "truncated_count": sum(bool(r.get("truncated")) for r in results),
            "pass_at_1": pass_at_k(results, 1)}


def sequence_metrics(sequences: Iterable[Iterable[int]], expert_count: int | None = None) -> dict:
    transitions = collections.Counter()
    run_lengths = []
    switches = tokens = pairs = 0
    local_entropies = []
    for raw in sequences:
        seq = [int(x) for x in raw]
        tokens += len(seq)
        pairs += max(len(seq) - 1, 0)
        for offset in range(0, max(len(seq) - 1, 0), 32):
            window = seq[offset:offset + 33]
            counts = collections.Counter(zip(window, window[1:]))
            total = sum(counts.values())
            if total:
                local_entropies.append(-sum((n / total) * math.log(n / total) for n in counts.values()))
        for a, b in zip(seq, seq[1:]):
            transitions[(a, b)] += 1
            switches += a != b
        if seq:
            last, run = seq[0], 1
            for value in seq[1:]:
                if value == last:
                    run += 1
                else:
                    run_lengths.append(run); last, run = value, 1
            run_lengths.append(run)
    matrix = [[0] * expert_count for _ in range(expert_count)] if expert_count is not None else None
    if matrix is not None:
        for (a, b), count in transitions.items():
            if 0 <= a < expert_count and 0 <= b < expert_count:
                matrix[a][b] += count
    transition_total = sum(transitions.values())
    transition_probs = [count / transition_total for count in transitions.values()] if transition_total else []
    transition_entropy = -sum(p * math.log(p) for p in transition_probs if p)
    return {"tokens": tokens, "switches": switches, "switch_rate": switches / pairs if pairs else None,
            "run_length_mean": sum(run_lengths) / len(run_lengths) if run_lengths else None,
            "transition_entropy_nats": transition_entropy if transition_total else None,
            "local_transition_entropy_32_nats": sum(local_entropies) / len(local_entropies) if local_entropies else None,
            "transition_counts": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())},
            "transition_matrix": matrix}


def routing_summary(records: Iterable[dict]) -> dict:
    """Aggregate per-layer selected expert IDs by category, language and source."""
    groups = collections.defaultdict(collections.Counter)
    transitions = collections.defaultdict(collections.Counter)
    configured_experts = collections.defaultdict(int)
    token_counts = collections.Counter()
    sequences = collections.defaultdict(list)
    previous = {}
    for record in records:
        category = record.get("category", record.get("kind", "unknown"))
        source = record.get("source_type", record.get("repository", record.get("source", "unknown")))
        group = "/".join(str(v) for v in (category, record.get("language", "unknown"), source))
        for layer, raw_ids in record.get("layers", {}).items():
            # Accept Top-1 `[expert, ...]` as well as Top-k
            # `[[expert0, expert1], ...]` traces. Load counts include every
            # selected slot; sequence transitions follow slot zero per token.
            ids = [int(expert) for item in raw_ids for expert in (item if isinstance(item, (list, tuple)) else [item])]
            sequence = [int(item[0] if isinstance(item, (list, tuple)) else item) for item in raw_ids]
            key = (group, str(layer))
            groups[key].update(ids)
            token_counts[key] += len(raw_ids)
            configured_experts[key] = max(configured_experts[key], int(record.get("expert_count", 0)))
            if key in previous:
                transitions[key].update(zip(previous[key], sequence))
            previous[key] = sequence
            sequences[key].append(sequence)
    out = {}
    for key, counter in groups.items():
        n = sum(counter.values()); max_id = max(counter, default=-1)
        # Trace formats may attach expert_count per record; use its observed
        # maximum, otherwise mark full-model min/unused counts unavailable.
        probs = [counter[i] / n for i in range(max_id + 1)] if n else []
        positive = [p for p in probs if p]
        entropy = -sum(p * math.log(p) for p in positive)
        expert_total = max(configured_experts[key], max_id + 1)
        full_counts = [counter[i] for i in range(expert_total)]
        full_probs = [count / n for count in full_counts] if n else []
        matrix = [[0] * expert_total for _ in range(expert_total)]
        for (a, b), count in transitions[key].items():
            if 0 <= a < expert_total and 0 <= b < expert_total:
                matrix[a][b] += count
        mean = n / max(expert_total, 1)
        out[f"{key[0]}/layer{key[1]}"] = {
            "tokens": n, "counts": dict(counter), "hard_entropy_nats": entropy,
            "top_k_slots_per_token": (n / token_counts[key]) if token_counts[key] else None,
            "load_cv": (math.sqrt(sum((c - mean) ** 2 for c in full_counts) / len(full_counts)) / mean) if mean and full_counts else None,
            "max_share": max(full_probs, default=0), "min_share": min(full_probs, default=0),
            "overloaded_share_over_2x_uniform": sum(p > (2 / expert_total) for p in full_probs) / expert_total if full_probs else None,
            "unused_experts": sum(c == 0 for c in full_counts),
            "full_expert_count": expert_total if configured_experts[key] else None, "confidence_margin": None,
            "unavailable_fields": (["Full-model unused/min load require configured expert count."] if not configured_experts[key] else []) + [
                                   "Router confidence margin requires logits/probabilities and is not present in selected-ID traces."],
            "transitions": {f"{a}->{b}": n for (a, b), n in sorted(transitions[key].items())},
            "transition_matrix": matrix if configured_experts[key] else None,
            "sequence_metrics": sequence_metrics(sequences[key], expert_total if configured_experts[key] else None)}
    return {"groups": out, "scope": "Selected expert IDs only; unobserved experts beyond the observed ID range require model metadata."}


def ngram_summary(statistics: dict | None) -> dict:
    if not statistics:
        return {"available": False, "reason": "No bounded n-gram statistics were supplied; unavailable is not zero."}
    banks = []
    for bank in statistics.get("banks", []):
        histogram = bank.get("lookup_frequency_histogram")
        percentiles = None
        if histogram:
            expanded = sorted((int(freq), int(count)) for freq, count in histogram.items())
            samples = [freq for freq, count in expanded for _ in range(count)]
            def percentile(q):
                return samples[min(int((len(samples) - 1) * q), len(samples) - 1)] if samples else None
            percentiles = {"p50": percentile(.50), "p90": percentile(.90), "p99": percentile(.99)}
        banks.append({"order": bank.get("order"), "rows": bank.get("rows"),
                      "utilized_rows": bank.get("utilized_rows"), "distinct_ngrams": bank.get("distinct_ngrams"),
                      "collisions": bank.get("collisions"), "collision_fraction": bank.get("collision_fraction"),
                      "lookup_frequency_histogram": histogram, "lookup_frequency_percentiles": percentiles,
                      "scope": "Bounded supplied sample only; no lifetime or corpus-wide estimate."})
    gate_fields = {key: statistics[key] for key in ("gate_mean", "gate_min", "gate_max", "gate_histogram10bins") if key in statistics}
    return {"available": bool(banks), "banks": banks, "gate": gate_fields or None,
            "scope": statistics.get("scope", "Reported input scope")}


def efficiency_summary(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    def values(key):
        return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
    fields = ("generation_seconds", "model_load_seconds", "peak_vram_bytes", "cpu_rss_bytes", "checkpoint_bytes",
              "generated_tokens", "tokens_per_second")
    summary = {}
    for field in fields:
        observed = values(field)
        summary[field] = {"observed_count": len(observed), "sum": sum(observed) if observed else None,
                          "max": max(observed) if observed else None}
    summary["scope"] = "Only metrics carried by source rows are summarized; missing fields remain unavailable, not zero."
    return summary


@contextlib.contextmanager
def evaluation_memory_ablation(memory):
    """Temporarily disable residual memory for inference and restore exact state."""
    if memory is None:
        yield
        return
    prior = memory.ablate
    if memory.training:
        raise RuntimeError("Memory ablation requires evaluation mode")
    memory.ablate = True
    try:
        yield
    finally:
        memory.ablate = prior


def checkpoint_candidates(root: Path) -> list[dict]:
    """Queue-only discovery. It never loads a checkpoint or launches evaluation."""
    candidates = []
    for path in sorted(Path(root).rglob("*.pt")):
        if path.name not in {"model_250M.pt", "model_500M.pt", "last.pt"}:
            continue
        digest_state = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest_state.update(chunk)
        digest = digest_state.hexdigest()
        candidates.append({"path": str(path.resolve()), "sha256": digest,
                           "status": "queued_manual_review", "gpu_used": False})
    return candidates


def architecture_estimate(*, vocab: int, d_model: int, layers: int, dense_ffn: int,
                           experts: int = 0, top_k: int = 0, moe_layers: int = 0,
                           ngram_rows: int = 0, ngram_banks: int = 0, ngram_dim: int = 0,
                           dtype_bytes: int = 2, optimizer_multiplier: float = 2.0) -> dict:
    """Transparent first-order parameter/storage estimate; excludes activations."""
    if min(vocab, d_model, layers, dense_ffn, dtype_bytes) <= 0 or optimizer_multiplier < 0:
        raise ValueError("Dimensions and dtype bytes must be positive; multiplier cannot be negative")
    if not 0 <= moe_layers <= layers or not 0 <= top_k <= experts:
        raise ValueError("MoE layers and top_k must fit the architecture")
    if min(ngram_rows, ngram_banks, ngram_dim) < 0:
        raise ValueError("N-gram dimensions cannot be negative")
    if moe_layers and (experts < 1 or top_k < 1):
        raise ValueError("MoE layers require at least one expert and a positive top_k")
    embedding = vocab * d_model
    attention = layers * 4 * d_model * d_model
    dense_block = layers * 2 * d_model * dense_ffn
    moe_total = moe_layers * experts * 2 * d_model * dense_ffn
    memory = ngram_rows * ngram_banks * ngram_dim + (ngram_banks * ngram_dim * d_model if ngram_banks else 0)
    stored = embedding + attention + dense_block * (layers - moe_layers) / layers + moe_total + memory
    dense_active = dense_block * (layers - moe_layers) / layers
    active = embedding + attention + dense_active + (moe_layers * top_k * 2 * d_model * dense_ffn) + memory
    return {"stored_parameters_estimate": int(stored), "active_parameters_per_token_estimate": int(active),
            "weight_bytes_estimate": int(stored * dtype_bytes),
            "optimizer_state_bytes_estimate": int(stored * 4 * optimizer_multiplier),
            "assumptions": ["Dense attention projection is approximated as four d_model squared matrices per layer.",
                            "MLP uses two d_model by FFN matrices; biases, norms, embeddings tying, activations, fragmentation and optimizer implementation overhead are omitted.",
                            "MoE replaces dense MLPs on moe_layers; active top_k experts are counted per token.",
                            "Optimizer-state bytes assume float32 state arrays multiplied by optimizer_multiplier (2 arrays is a common Adam first-order estimate).",
                            "These parameter/storage estimates do not predict quality or declare a winner."]}

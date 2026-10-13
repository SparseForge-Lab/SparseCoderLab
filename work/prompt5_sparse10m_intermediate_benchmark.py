"""Matched 25-prompt x 7-temperature diagnostic at a verified intermediate checkpoint."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\SparseCoderLab")
TOKENS = 466_944_000
VARIANT = "sparse75_ngram10m"
EXPECTED_TOKENIZER = "832daf453fc64e28f50df00ac2e458718576bb26e3f16e737f18b120ae4e309a"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer

BASELINE_SCRIPT = ROOT / "work" / "prompt5_dense75_250m_fullgpu_test.py"
BASELINE_SPEC = importlib.util.spec_from_file_location("prompt5_intermediate_baseline", BASELINE_SCRIPT)
BASE = importlib.util.module_from_spec(BASELINE_SPEC)
assert BASELINE_SPEC and BASELINE_SPEC.loader
BASELINE_SPEC.loader.exec_module(BASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-compute-pid", type=int, action="append", default=[])
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5070":
        raise RuntimeError("Expected the authorized RTX 5070")
    # Under WDDM, --query-compute-apps also reports graphics clients as
    # compute apps. Read the process table and reject CUDA-only (Type C)
    # workers while allowing normal C+G desktop/game clients such as Roblox.
    process_table = os.popen("nvidia-smi").read().splitlines()
    active = []
    for line in process_table:
        match = re.match(r"^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(\S+)\s+", line)
        if match and match.group(2) == "C":
            active.append(match.group(1))
    allowed = set(args.allow_compute_pid) | {os.getpid()}
    competitors = sorted({int(value.strip()) for value in active if value.strip().isdigit()} - allowed)
    if competitors:
        raise RuntimeError(f"A conflicting CUDA compute process is active: {competitors}")

    run = ROOT / "experiments" / "prompt5_500M_v1" / VARIANT
    receipt = json.loads((run / "checkpoint_verification.json").read_text(encoding="utf-8"))
    checkpoint_record = receipt.get("latest") or {}
    checkpoint = ROOT / checkpoint_record.get("path", "")
    if (checkpoint_record.get("tokens_seen") != TOKENS or checkpoint_record.get("step") != TOKENS // 8192
            or not checkpoint.is_file() or sha256(checkpoint) != checkpoint_record.get("sha256")):
        raise RuntimeError("The exact intermediate checkpoint does not match its verified receipt")

    baseline_run = ROOT / "experiments" / "prompt5_v1" / VARIANT
    config = json.loads((baseline_run / "config.json").read_text(encoding="utf-8"))
    tokenizer_path = ROOT / config["data"]["tokenizer"]
    tokenizer_hash = sha256(tokenizer_path)
    if tokenizer_hash != EXPECTED_TOKENIZER:
        raise RuntimeError("Frozen tokenizer identity changed")

    output_dir = ROOT / "results" / "prompt5_500M_v1" / "milestone_benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{VARIANT}_{TOKENS}_tokens_intermediate_fullgpu_benchmark"
    jsonl = output_dir / f"{stem}.jsonl"
    detail = output_dir / f"{stem}.txt"
    existing = {}
    if jsonl.exists():
        for line_no, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["prompt_number"]), float(row["temperature"]))
            if (key in existing or row.get("checkpoint_sha256") != checkpoint_record["sha256"]
                    or row.get("training_tokens") != TOKENS):
                raise ValueError(f"Existing intermediate row is duplicated or mismatched (line {line_no})")
            existing[key] = row

    expected = {(p, t) for t in BASE.TEMPERATURES for p in range(1, len(BASE.PROMPTS) + 1)}
    if not set(existing).issubset(expected):
        raise ValueError("Unexpected benchmark slot in saved intermediate output")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = BASE.LanguageModel(config).to(BASE.DEVICE).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("metadata", {}).get("tokens_seen") != TOKENS:
        raise RuntimeError("Checkpoint metadata does not bind the requested token count")
    model.load_state_dict(state["model"], strict=True)
    del state
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("The intermediate model is not resident on CUDA")

    baseline_jsonl = output_dir / f"{VARIANT}_375M_fullgpu_benchmark.jsonl"
    baseline_rows = {}
    for line in baseline_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            baseline_rows[(int(row["prompt_number"]), float(row["temperature"]))] = row
    if set(baseline_rows) != expected:
        raise RuntimeError("The matched 375M comparison report is incomplete")

    started = time.perf_counter()
    with jsonl.open("a", encoding="utf-8", newline="\n") as stream:
        for temperature in BASE.TEMPERATURES:
            for prompt_number, prompt in enumerate(BASE.PROMPTS, 1):
                key = (prompt_number, temperature)
                if key in existing:
                    continue
                generated = BASE.generation(model, tokenizer, prompt, temperature)
                row = {
                    "model": f"Sparse 75M + 10M Ngram @ {TOKENS} tokens",
                    "variant": VARIANT,
                    "stage": f"{TOKENS}-token-intermediate",
                    "training_tokens": TOKENS,
                    "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                    "checkpoint_sha256": checkpoint_record["sha256"],
                    "tokenizer_sha256": tokenizer_hash,
                    "device": str(BASE.DEVICE),
                    "prompt_number": prompt_number,
                    "input": prompt,
                    "temperature": temperature,
                    "generation_seed": BASE.SEED,
                    "max_new_tokens": BASE.MAX_NEW_TOKENS,
                    "top_p": BASE.TOP_P,
                    "top_k": BASE.TOP_K if temperature != 0 else None,
                    "repetition_penalty": BASE.REPETITION_PENALTY,
                    "stop_tokens": ["<eos>", "<doc>"],
                    "raw_output": generated["output"],
                    "generation_seconds": generated["generation_seconds"],
                    "generated_token_count": generated["generated_tokens"],
                    "tokens_per_second": generated["tokens_per_second"],
                    "prompt_truncated": generated["prompt_truncated"],
                    "heuristics": BASE.indicators(generated["output"], prompt),
                    "benchmark_source_sha256": sha256(BASELINE_SCRIPT),
                    "orchestration_sha256": sha256(Path(__file__)),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                existing[key] = row
                if len(existing) % 10 == 0:
                    print(json.dumps({"variant": VARIANT, "tokens": TOKENS,
                                      "completed": len(existing), "total": len(expected)}), flush=True)

    if set(existing) != expected:
        raise RuntimeError(f"Intermediate benchmark incomplete: {len(existing)}/{len(expected)}")
    if sha256(checkpoint) != checkpoint_record["sha256"] or sha256(tokenizer_path) != tokenizer_hash:
        raise RuntimeError("Benchmark inputs changed while running")

    heuristic_keys = ("python_like_code", "contains_return", "parseable_as_python_with_prompt",
                      "obvious_repetition_loop", "empty", "url", "markdown_heading")
    lines = [
        "Matched intermediate 25-prompt x 7-temperature GPU diagnostic",
        f"Variant: {VARIANT}", f"Training tokens: {TOKENS:,}",
        f"Checkpoint: {checkpoint.relative_to(ROOT).as_posix()}",
        f"Checkpoint SHA-256: {checkpoint_record['sha256']}",
        f"Tokenizer SHA-256: {tokenizer_hash}",
        "Settings: frozen Prompt-5 prompt set, seed, temperatures, top-k, stop tokens, and 128-token cap.",
        "These regex/AST indicators are approximate; generated code was not executed.",
        f"Generation wall time: {time.perf_counter() - started:.2f} seconds",
        "", "Comparison with the matched 375M checkpoint:",
    ]
    for temperature in BASE.TEMPERATURES:
        current = [row for row in existing.values() if float(row["temperature"]) == temperature]
        earlier = [row for row in baseline_rows.values() if float(row["temperature"]) == temperature]
        counts = {key: sum(int(row["heuristics"].get(key, 0)) for row in current) for key in heuristic_keys}
        earlier_counts = {key: sum(int(row["heuristics"].get(key, 0)) for row in earlier) for key in heuristic_keys}
        deltas = {key: counts[key] - earlier_counts[key] for key in heuristic_keys}
        lines.append(f"Temperature {temperature:.1f}: current={counts}; change_vs_375M={deltas}")

    detail_contents = "\n".join(lines) + "\n"
    if detail.exists():
        prior_report = detail.read_text(encoding="utf-8")
        stable_prior = re.sub(r"Generation wall time: [0-9.]+ seconds", "Generation wall time: <recorded>", prior_report)
        stable_current = re.sub(r"Generation wall time: [0-9.]+ seconds", "Generation wall time: <recorded>", detail_contents)
        if stable_prior != stable_current:
            raise FileExistsError(f"Preserving an existing, different report: {detail}")
    else:
        temporary = detail.with_suffix(".txt.tmp")
        temporary.write_text(detail_contents, encoding="utf-8", newline="\n")
        os.replace(temporary, detail)
    print(json.dumps({"variant": VARIANT, "training_tokens": TOKENS, "rows": len(existing),
                      "jsonl": str(jsonl)}), flush=True)


if __name__ == "__main__":
    main()

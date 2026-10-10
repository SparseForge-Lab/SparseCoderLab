"""Generate the frozen 100-prompt functional suite from matched 250M gates."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import psutil
import torch
from tokenizers import Tokenizer

from src.utils.hashing import sha256_file
from tools.evaluate_repository_gate import MANIFEST, entry_model
from tools.generate_canonical_python import generate_batch, gpu_allowed


BENCHMARK = Path("configs/eval/unified_python_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/unified_python_250M_greedy_r1.jsonl"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    efficiency_path = args.output.with_suffix(".efficiency.jsonl")
    benchmark_path = args.benchmark
    if args.output.exists() or efficiency_path.exists():
        raise FileExistsError("Preserve existing generations/receipts; pass a fresh --output")
    if not 1 <= args.batch_size <= 8:
        raise ValueError("Batch size must be between 1 and 8")
    if not 1 <= args.max_new_tokens <= 1024:
        raise ValueError("max_new_tokens must be between 1 and 1024")
    manifest_path = args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError("Matched 250M checkpoint manifest is not ready")
    if Path(".local/process_controls/repository_campaign.lock").exists():
        raise RuntimeError("Training campaign lock is active; GPU evaluation remains queued")
    gpu_allowed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != "NVIDIA GeForce RTX 5070":
        raise RuntimeError(f"Unexpected GPU: {gpu_name}")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    benchmark_raw = benchmark_path.read_bytes()
    benchmark = json.loads(benchmark_raw)
    if benchmark.get("benchmark_id") != "unified_python_v1" or len(benchmark.get("tasks", [])) != 100:
        raise ValueError("Expected the frozen 100-prompt unified benchmark")
    manifest_raw = manifest_path.read_bytes()
    manifest_sha256 = sha256_file(manifest_path)
    manifest_metadata = json.loads(manifest_raw)
    entries = json.loads(manifest_raw)["matched_gate_checkpoints"]
    if len(entries) != 4 or any(entry["total_tokens"] != 250_003_456 for entry in entries.values()):
        raise ValueError("Expected all four completed matched 75M/250M-token checkpoints")
    tokenizer_path = Path("data/research_v1/tokenizer.json")
    tokenizer_hash = sha256_file(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    generation_git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    generation_git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8", newline="\n") as handle, efficiency_path.open("x", encoding="utf8", newline="\n") as efficiency_handle:
        for variant, entry in entries.items():
            gpu_allowed()
            model_load_started = time.perf_counter()
            model, cfg, checkpoint = entry_model(variant, entry)
            if sha256_file(Path(cfg["data"]["tokenizer"])) != tokenizer_hash:
                raise ValueError(f"Generation tokenizer differs from matched config for {variant}")
            torch.cuda.synchronize()
            model_load_seconds = time.perf_counter() - model_load_started
            checkpoint_bytes = checkpoint.stat().st_size
            torch.cuda.reset_peak_memory_stats()
            generation_started = time.perf_counter()
            generated_tokens = 0
            generated_count = 0
            for offset in range(0, len(benchmark["tasks"]), args.batch_size):
                gpu_allowed()
                tasks = benchmark["tasks"][offset:offset + args.batch_size]
                generated = generate_batch(model, tokenizer, [task["prompt"] for task in tasks], 0.0,
                                           maximum=args.max_new_tokens)
                for task, (completion, token_count) in zip(tasks, generated):
                    row = {"variant": variant, "temperature": 0.0, "task_id": task["id"],
                           "family_id": task["family_id"], "category": task["category"],
                           "paraphrase_index": task["paraphrase_index"], "prompt_number": task["prompt_number"],
                           "input": task["prompt"], "raw_output": completion,
                           "checkpoint": checkpoint.as_posix(), "checkpoint_sha256": entry["model_weights"]["sha256"],
                           "checkpoint_bytes": checkpoint_bytes, "training_tokens": entry["total_tokens"],
                           "checkpoint_manifest_sha256": manifest_sha256,
                           "checkpoint_manifest_source_sha256": manifest_metadata.get("source_manifest_sha256"),
                           "generation_seed": 42, "max_new_tokens": args.max_new_tokens, "top_p": None, "top_k": None,
                           "decoding": "greedy",
                           "repetition_penalty": 1.0, "stop_tokens": ["<eos>", "<doc>"],
                           "device": "cuda", "bf16": True, "tf32": True, "generation_batch_size": args.batch_size,
                           "gpu_name": gpu_name,
                           "deterministic_algorithms": True,
                           "generated_tokens": token_count, "benchmark_file": benchmark_path.as_posix(),
                           "benchmark_sha256": sha256_file(benchmark_path),
                           "tokenizer_sha256": tokenizer_hash,
                           "sampler_sha256": sha256_file(Path("tools/generate_canonical_python.py")),
                           "generation_git_commit": generation_git_commit,
                           "generation_git_dirty": generation_git_dirty,
                           "orchestration_sha256": sha256_file(Path(__file__))}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    generated_tokens += token_count
                    generated_count += 1
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_started
            receipt = {"record_type": "efficiency", "variant": variant,
                       "generation_seconds": generation_seconds, "generated_tokens": generated_tokens,
                       "tokens_per_second": generated_tokens / generation_seconds if generation_seconds else None,
                       "model_load_seconds": model_load_seconds,
                       "gpu_name": gpu_name,
                       "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                       "cpu_rss_bytes": psutil.Process().memory_info().rss,
                       "checkpoint_bytes": checkpoint_bytes, "generation_count": generated_count}
            efficiency_handle.write(json.dumps(receipt) + "\n")
            efficiency_handle.flush()
            if sha256_file(checkpoint) != entry["model_weights"]["sha256"]:
                raise ValueError("Checkpoint changed during generation")
            print(json.dumps({"variant": variant, "completed": generated_count,
                              "tokens_per_second": receipt["tokens_per_second"], "gpu_used": True}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    if benchmark_path.read_bytes() != benchmark_raw or manifest_path.read_bytes() != manifest_raw:
        raise ValueError("Benchmark or checkpoint manifest changed during generation")


if __name__ == "__main__":
    main()

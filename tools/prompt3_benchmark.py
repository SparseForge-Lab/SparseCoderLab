"""Disposable steady-state GPU throughput sweep for the Prompt-3 candidates."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch

from src.config import load_config
from src.model import LanguageModel
from src.training.data import PackedStream
from src.training.engine import amp, make_optimizer, require_cuda, seed_all
from tools.shard_data import verify_shards

VARIANTS = (
    "dense75_ref",
    "sparse75",
    "sparse75_ngram10m",
    "sparse75_ngram25m",
)


def trial(tag: str, microbatch: int, warmup: int, steps: int, repetition: int) -> dict:
    cfg = load_config(f"configs/prompt3/{tag}.yaml")
    training = cfg["training"]
    if 8 % microbatch:
        raise ValueError("Microbatch must divide the frozen 8-sequence effective batch")
    training["microbatch"] = microbatch
    training["accumulation"] = 8 // microbatch
    seed_all(421337)
    model = LanguageModel(cfg).cuda()
    optimizer, scheduler = make_optimizer(model, cfg)
    stream = PackedStream(Path(cfg["data"]["shards"]), "train", training["context"], 42)
    torch.backends.cuda.matmul.allow_tf32 = training["tf32"]
    torch.backends.cudnn.allow_tf32 = training["tf32"]
    torch.cuda.reset_peak_memory_stats()
    observations = []
    try:
        model.train()
        for index in range(warmup + steps):
            started = time.perf_counter()
            data_seconds = 0.0
            optimizer.zero_grad(set_to_none=True)
            for _ in range(training["accumulation"]):
                data_started = time.perf_counter()
                x, y = stream.next(microbatch, "cuda")
                data_seconds += time.perf_counter() - data_started
                with amp(cfg):
                    output = model(x, y)
                    loss = output["loss"] / training["accumulation"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss for {tag}/{microbatch}")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - started
            if index >= warmup:
                observations.append({"step_seconds": step_seconds, "data_seconds": data_seconds})
        tokens_per_step = training["context"] * microbatch * training["accumulation"]
        mean_step = sum(row["step_seconds"] for row in observations) / len(observations)
        mean_data = sum(row["data_seconds"] for row in observations) / len(observations)
        return {
            "model": tag,
            "microbatch": microbatch,
            "accumulation": training["accumulation"],
            "repetition": repetition,
            "status": "measured",
            "effective_tokens_per_update": tokens_per_step,
            "context": training["context"],
            "warmup_updates": warmup,
            "measured_updates": steps,
            "mean_step_seconds": mean_step,
            "wall_tokens_per_second": tokens_per_step / mean_step,
            "mean_dataloader_seconds_per_update": mean_data,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "scope": "Disposable BF16 training updates on frozen corpus order; excludes initialization/evaluation/checkpoint writes. No quality claim.",
        }
    finally:
        del model, optimizer, scheduler, stream
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Execute the disposable GPU benchmark")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--microbatches", nargs="+", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Pass --run to execute; this is a GPU benchmark, not training")
    require_cuda("cuda")
    cfg = load_config(f"configs/prompt3/{VARIANTS[0]}.yaml")
    verify_shards(cfg)
    rows = []
    for tag in VARIANTS:
        for microbatch in args.microbatches:
            if microbatch not in (1, 2, 4, 8):
                continue
            for repetition in range(1, args.repetitions + 1):
                try:
                    row = trial(tag, microbatch, args.warmup, args.steps, repetition)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row = {"model": tag, "microbatch": microbatch, "accumulation": 8 // microbatch,
                           "repetition": repetition, "status": "oom", "effective_tokens_per_update": 8192,
                           "context": 1024, "scope": "Disposable throughput trial exceeded available VRAM."}
                rows.append(row)
                print(json.dumps(row), flush=True)
    destination = Path("results/prompt3/throughput")
    destination.mkdir(parents=True, exist_ok=True)
    report = {"status": "complete", "warmup_updates": args.warmup, "measured_updates_per_case": args.steps,
              "repetitions": args.repetitions, "rows": rows,
              "selection_rule": "Choose one shared microbatch/accumulation setting maximizing median wall tokens/s across all four candidates, subject to no OOM and the 15% VRAM headroom target."}
    (destination / "measurements.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if rows:
        with (destination / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"saved": str(destination), "rows": len(rows)}))


if __name__ == "__main__":
    main()

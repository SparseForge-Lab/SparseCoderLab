"""Run one evidence-gated Prompt-3 cumulative token stage."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch

from src.config import fingerprint, load_config
from src.eval.research import evaluate, freeze_evaluation
from src.model import LanguageModel
from src.training.engine import require_cuda, run
from tools.prompt3_preflight import VARIANTS, source_hash

GATES = {"100M": 100_007_936, "180M": 180_002_816, "250M": 250_003_456, "500M": 500_006_912, "1B": 1_000_005_632}
ROOT = Path("results/prompt3")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temp)
    os.replace(temp, target)


def check_gate() -> dict:
    gate_path = ROOT / "readiness_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("passed") or gate.get("source_hash") != source_hash():
        raise RuntimeError("Prompt-3 blocked: current source does not match the passing readiness gate")
    disk = shutil.disk_usage(Path.cwd())
    if disk.free < 5 * 1024**3:
        raise RuntimeError(f"Prompt-3 blocked: only {disk.free / 1024**3:.1f} GiB disk space remains")
    project_bytes = 0
    for base, dirs, files in os.walk(Path.cwd()):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}]
        for name in files:
            try:
                project_bytes += (Path(base) / name).stat().st_size
            except OSError:
                pass
    project_limit = 45 * 1024**3
    if project_bytes > project_limit:
        raise RuntimeError(f"Prompt-3 blocked: project bytes exceed 45 GiB cap ({project_bytes / 1024**3:.1f} GiB)")
    return gate


def run_variant(tag: str, endpoint_name: str, endpoint: int, wall_minutes: float) -> dict:
    cfg_path = Path("configs/prompt3") / f"{tag}.yaml"
    cfg = load_config(cfg_path)
    gate = json.loads((ROOT / "readiness_gate.json").read_text(encoding="utf-8"))
    if gate["config_fingerprints"][tag] != fingerprint(cfg):
        raise RuntimeError(f"Prompt-3 config changed after preflight: {tag}")
    run_dir = Path("experiments/prompt3_v1") / tag
    ckpt = run_dir / "checkpoints" / "last.pt"
    resume = ckpt if ckpt.exists() else None
    print(f"START {tag} -> {endpoint_name} ({endpoint:,} cumulative tokens); resume={resume is not None}", flush=True)
    result = run(cfg, run_dir, max_wall_minutes=wall_minutes, target_tokens=endpoint, resume=resume)
    if result["tokens_seen"] < endpoint:
        print(f"PAUSED {tag}: {result['tokens_seen']:,}/{endpoint:,} tokens; resume checkpoint saved", flush=True)
        return {"tag": tag, "complete": False, "tokens_seen": result["tokens_seen"], "target": endpoint}
    atomic_copy(ckpt, run_dir / "checkpoints" / f"latest_verified_{endpoint_name}.pt")

    # Preserve an immutable, model-only gate snapshot while keeping full Adam
    # state only for the rolling and newest verified resume checkpoints.
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    milestone = run_dir / "checkpoints" / f"model_{endpoint_name}.pt"
    temp = milestone.with_suffix(".tmp")
    torch.save(state["model"], temp)
    os.replace(temp, milestone)
    del state

    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["training"]["microbatch"] = 2
    index = freeze_evaluation(eval_cfg)
    model = LanguageModel(eval_cfg).cuda()
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    del saved
    start = time.perf_counter()
    normal = evaluate(model, eval_cfg, index, full=True, routes=True)
    ablated = None
    if model.memory is not None:
        model.memory.ablate = True
        try:
            ablated = evaluate(model, eval_cfg, index, full=True, routes=False)
        finally:
            model.memory.ablate = False
    eval_seconds = time.perf_counter() - start
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    report = {
        "variant": tag, "gate": endpoint_name, "target_tokens": endpoint,
        "actual_tokens": result["tokens_seen"], "seed": cfg["training"]["seed"],
        "stored_params": summary["stored_params"], "active_params_est": summary["active_params_est"],
        "training_summary": summary, "evaluation": normal,
        "ngram_residual_off_ablation": ablated,
        "ablation_delta_ablated_minus_normal": ({k: ablated[k] - normal[k] for k in normal
            if isinstance(normal.get(k), (int, float)) and ablated.get(k) is not None} if ablated else None),
        "evaluation_seconds": eval_seconds, "evaluation_microbatch": 2,
        "checkpoint_sha256": sha(ckpt), "model_snapshot_sha256": sha(milestone),
        "config_sha256": sha(cfg_path), "config_fingerprint": fingerprint(cfg),
        "data_manifest_sha256": sha(Path(cfg["data"]["shards"]) / "manifest.json"),
        "tokenizer_sha256": sha(Path(cfg["data"]["tokenizer"])),
        "source_hash": source_hash(), "scope": "Single-seed internal proxy evaluation; not a public benchmark.",
    }
    report_path = run_dir / f"summary_{endpoint_name}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    print(f"COMPLETE {tag} {endpoint_name}: tokens={result['tokens_seen']:,}, val_loss={normal['val_loss']:.6f}, eval={eval_seconds/60:.1f}min", flush=True)
    return {"tag": tag, "complete": True, "tokens_seen": result["tokens_seen"],
            "val_loss": normal["val_loss"], "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", choices=GATES, default="100M")
    parser.add_argument("--models", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--max-wall-minutes", type=float, default=300)
    args = parser.parse_args()
    if args.endpoint == "180M" and args.models != ["sparse75_ngram25m"]:
        raise ValueError("The supplemental 180M continuation is limited to sparse75_ngram25m")
    require_cuda("cuda")
    check_gate()
    rows = []
    for tag in args.models:
        row = run_variant(tag, args.endpoint, GATES[args.endpoint], args.max_wall_minutes)
        rows.append(row)
        if not row["complete"]:
            break
    result = {"endpoint": args.endpoint, "results": rows, "source_hash": source_hash(),
              "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = ROOT / f"stage_{args.endpoint}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

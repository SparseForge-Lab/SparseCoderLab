"""Verify immutable Prompt-3 transition and historical checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.config import fingerprint, load_config
from src.model import LanguageModel
from tools.count_params import count_model

ROOT = Path("experiments/prompt3_v1")
VARIANTS = {
    "dense75_ref": "dense75_ref",
    "sparse75": "sparse75",
    "sparse75_ngram10m": "sparse75_ngram10m",
    "sparse75_ngram25m": "sparse75_ngram25m",
}
CANONICAL_TOKENS = 100_007_936


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, root: Path) -> dict:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def metadata(state: dict) -> dict:
    value = state.get("metadata")
    if not isinstance(value, dict):
        raise ValueError("training checkpoint has no metadata object")
    return value


def verify_snapshot(model: LanguageModel, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError(f"expected a model-only state dict: {path}")
    model.load_state_dict(state, strict=True)


def verify_transition(tag: str, config_name: str, project_root: Path) -> dict:
    cfg_path = project_root / "configs" / "prompt3" / f"{config_name}.yaml"
    cfg = load_config(cfg_path)
    run_dir = project_root / ROOT / tag
    resume_path = run_dir / "checkpoints" / "latest_verified_100M.pt"
    snapshot_path = run_dir / "checkpoints" / "model_100M.pt"
    rolling_path = run_dir / "checkpoints" / "last.pt"
    full = torch.load(resume_path, map_location="cpu", weights_only=False)
    meta = metadata(full)
    if meta.get("tokens_seen") != CANONICAL_TOKENS:
        raise ValueError(f"{tag} canonical token count is {meta.get('tokens_seen')}, expected {CANONICAL_TOKENS}")
    if meta.get("config_hash") != fingerprint(cfg):
        raise ValueError(f"{tag} checkpoint/config fingerprint mismatch")

    model = LanguageModel(cfg)
    model.load_state_dict(full["model"], strict=True)
    verify_snapshot(model, snapshot_path)
    counts = count_model(model)
    summary = json.loads((run_dir / "summary_100M.json").read_text(encoding="utf-8"))
    if counts["total"] != summary["stored_params"] or counts["active_estimate"] != summary["active_params_est"]:
        raise ValueError(f"{tag} parameter counts disagree with the 100M evaluation summary")
    expected_memory = cfg["memory"]["banks"] * cfg["memory"]["rows"] * cfg["memory"]["dim"] if cfg["memory"]["enabled"] else 0
    if counts["memory_tables"] != expected_memory:
        raise ValueError(f"{tag} conditional-memory capacity mismatch")
    if not all(key in full for key in ("optimizer", "scheduler", "rng")):
        raise ValueError(f"{tag} full-state checkpoint is missing optimizer/scheduler/RNG state")
    if not all(key in full["rng"] for key in ("python", "numpy", "torch", "cuda")):
        raise ValueError(f"{tag} full-state checkpoint is missing an RNG stream")

    current = torch.load(rolling_path, map_location="cpu", weights_only=False)
    current_meta = metadata(current)
    if current_meta.get("data_hash") != meta.get("data_hash") or current_meta.get("tokenizer_hash") != meta.get("tokenizer_hash"):
        raise ValueError(f"{tag} rolling checkpoint changed data/tokenizer identity")
    result = {
        "status": "verified",
        "canonical_transition_tokens": CANONICAL_TOKENS,
        "stored_parameters": counts["total"],
        "active_parameters_estimate": counts["active_estimate"],
        "ngram_table_parameters": counts["memory_tables"],
        "config_fingerprint": fingerprint(cfg),
        "data_manifest_sha256": meta["data_hash"],
        "tokenizer_sha256": meta["tokenizer_hash"],
        "config": identity(cfg_path, project_root),
        "model_weights": identity(snapshot_path, project_root),
        "full_resume_state": identity(resume_path, project_root),
        "resume_state": {"optimizer": True, "scheduler": True, "python_rng": True, "numpy_rng": True, "torch_rng": True, "cuda_rng": bool(full["rng"]["cuda"])},
        "latest_rolling_checkpoint": {
            **identity(rolling_path, project_root),
            "tokens_seen": current_meta["tokens_seen"],
            "step": current_meta["step"],
            "classification": "canonical-100M" if current_meta["tokens_seen"] == CANONICAL_TOKENS else "historical-noncanonical",
        },
    }
    del current, full, model
    return result


def verify_historical(tag: str, filename: str, project_root: Path) -> dict:
    path = project_root / ROOT / tag / "checkpoints" / filename
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "metadata" in state:
        meta = metadata(state)
        tokens = meta["tokens_seen"]
        step = meta["step"]
        config_hash = meta["config_hash"]
        optimizer, scheduler, rng = "optimizer" in state, "scheduler" in state, "rng" in state
    else:
        # The immutable model-only snapshot has no optimizer metadata. Bind its
        # token count and config identity to the adjacent full-state report.
        cfg = load_config(project_root / "configs" / "prompt3" / f"{tag}.yaml")
        model = LanguageModel(cfg)
        model.load_state_dict(state, strict=True)
        counts = count_model(model)
        gate_name = filename.removeprefix("model_").removesuffix(".pt")
        report = json.loads((project_root / ROOT / tag / f"summary_{gate_name}.json").read_text(encoding="utf-8"))
        if counts["total"] != report["stored_params"] or counts["active_estimate"] != report["active_params_est"]:
            raise ValueError(f"{tag} {gate_name} historical snapshot parameter mismatch")
        tokens = report["actual_tokens"]
        step = None
        config_hash = fingerprint(cfg)
        optimizer = scheduler = rng = False
        del model
    result = {
        **identity(path, project_root),
        "tokens_seen": tokens,
        "step": step,
        "config_fingerprint": config_hash,
        "classification": "historical-noncanonical",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "rng": rng,
    }
    del state
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/prompt3/transition_checkpoint_manifest.json")
    args = parser.parse_args()
    project_root = Path.cwd()
    canonical = {tag: verify_transition(tag, config, project_root) for tag, config in VARIANTS.items()}
    historical = {
        "dense75_ref": [
            verify_historical("dense75_ref", "latest_verified_250M.pt", project_root),
            verify_historical("dense75_ref", "model_250M.pt", project_root),
            verify_historical("dense75_ref", "last.pt", project_root),
        ],
        "sparse75": [verify_historical("sparse75", "last.pt", project_root)],
    }
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Freeze the common 100M model weights as the transition point; retain later partial continuations as historical evidence.",
        "canonical_rule": "Use only the 100M model_weights/full_resume_state listed here for the next real-data phase. Do not overwrite or promote later partial checkpoints into canonical starts.",
        "canonical_transition_checkpoints": canonical,
        "historical_noncanonical_checkpoints": historical,
        "integrity": "Every listed state was loaded from CPU; canonical full/model-only weights loaded with strict=True into the matching configured model.",
    }
    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": out_path.as_posix(), "variants_verified": len(canonical),
                      "historical_checkpoints": sum(len(rows) for rows in historical.values())}, indent=2))


if __name__ == "__main__":
    main()

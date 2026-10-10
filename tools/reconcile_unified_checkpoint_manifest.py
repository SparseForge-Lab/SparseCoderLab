"""Create a study-local attestation when config JSON formatting hashes drift."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


VARIANTS = ("dense75_ref", "sparse75", "sparse75_ngram10m", "sparse75_ngram25m")
TOKENS = 250_003_456


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/repository_training_v1/matched_gate_checkpoints.json"))
    parser.add_argument("--output", type=Path, default=Path("results/unified_python_250M_checkpoint_manifest_r1.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve existing receipt: {args.output}")

    source_raw = args.input.read_bytes()
    source_sha = hashlib.sha256(source_raw).hexdigest()
    manifest = json.loads(source_raw)
    entries = manifest.get("matched_gate_checkpoints", {})
    if set(entries) != set(VARIANTS):
        raise ValueError("Expected exactly the four matched Prompt-5 variants")

    reconciliations = {}
    for variant in VARIANTS:
        entry = entries[variant]
        if entry.get("total_tokens") != TOKENS:
            raise ValueError(f"Unexpected token stage for {variant}")
        config_path = Path(entry["config"]["path"])
        checkpoint_path = Path(entry["model_weights"]["path"])
        config = json.loads(config_path.read_text(encoding="utf8"))
        semantic_fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        recorded_fingerprint = entry["config"].get("fingerprint")
        summary = json.loads((Path("experiments/prompt5_v1") / variant / "campaign_summary.json").read_text(encoding="utf8"))
        if semantic_fingerprint != recorded_fingerprint or semantic_fingerprint != summary.get("config_hash"):
            raise ValueError(f"Config contents do not match the gate and campaign fingerprints for {variant}")
        checkpoint_sha = sha256(checkpoint_path)
        if checkpoint_sha != entry["model_weights"]["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch for {variant}")
        if checkpoint_sha != summary["model_only"]["sha256"]:
            raise ValueError(f"Campaign checkpoint SHA-256 mismatch for {variant}")
        actual_config_sha = sha256(config_path)
        previous_config_sha = entry["config"]["sha256"]
        if actual_config_sha != previous_config_sha:
            reconciliations[variant] = {"previous_sha256": previous_config_sha,
                                        "current_sha256": actual_config_sha,
                                        "semantic_fingerprint": semantic_fingerprint}
            entry["config"]["sha256"] = actual_config_sha

    manifest["source_manifest_sha256"] = source_sha
    manifest["config_hash_reconciliation"] = reconciliations
    manifest["reconciled_on_git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "source_manifest_sha256": source_sha, "reconciliations": reconciliations}, indent=2))


if __name__ == "__main__":
    main()

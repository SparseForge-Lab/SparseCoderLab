"""Fresh Prompt-3 source/data gates and four-model GPU resume-parity smoke."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from src.config import fingerprint, load_config
from src.model import LanguageModel
from src.training.checkpoint import resume_training, save_training
from src.training.data import PackedStream
from src.training.engine import amp, make_optimizer, require_cuda, seed_all
from tools.shard_data import verify_shards

VARIANTS = ("dense75_ref", "sparse75", "sparse75_ngram10m", "sparse75_ngram25m")
ROOT = Path("results/prompt3")
OPTIMIZER_RESUME_ATOL = 1e-5


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hash() -> str:
    paths = sorted(Path("src").rglob("*.py"))
    paths += sorted(Path("tools").glob("prompt3_*.py"))
    paths += sorted(Path("tests").glob("*.py"))
    paths += sorted(Path("configs/prompt3").glob("*.yaml"))
    return fingerprint({p.as_posix(): sha(p) for p in paths})


def junit_summary(path: Path) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    count = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    return count, failures


def optimizer_snapshot(optimizer: torch.optim.Optimizer) -> list[dict]:
    result = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            result.append({key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                           for key, value in state.items()})
    return result


def compare_optimizer(expected: list[dict], actual: list[dict]) -> tuple[float, bool]:
    if len(expected) != len(actual):
        return float("inf"), False
    maximum = 0.0
    for a, b in zip(expected, actual):
        if set(a) != set(b):
            return float("inf"), False
        for key in a:
            va, vb = a[key], b[key]
            if torch.is_tensor(va):
                if va.shape != vb.shape:
                    return float("inf"), False
                if va.is_floating_point():
                    maximum = max(maximum, float((va - vb).abs().max()) if va.numel() else 0.0)
                elif not torch.equal(va, vb):
                    return float("inf"), False
            elif va != vb:
                return float("inf"), False
    # CUDA reductions can vary slightly across the checkpoint boundary even
    # when the resumed loss and model weights agree. Record the actual delta
    # and use a tolerance well below the scale of Adam's optimizer buffers.
    return maximum, maximum <= OPTIMIZER_RESUME_ATOL


def train_step(model, optimizer, scheduler, cfg, x, y) -> float:
    optimizer.zero_grad(set_to_none=True)
    with amp(cfg):
        out = model(x, y)
        loss = out["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("Prompt-3 smoke produced nonfinite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    return float(loss.detach())


def gpu_resume_smoke(tag: str) -> dict:
    cfg = load_config(f"configs/prompt3/{tag}.yaml")
    seed_all(20261007)
    model = LanguageModel(cfg).cuda()
    optimizer, scheduler = make_optimizer(model, cfg)
    training = cfg["training"]
    stream = PackedStream(Path(cfg["data"]["shards"]), "train", training["context"], 42)
    x1, y1 = stream.next(training["microbatch"], "cuda")
    loss1 = train_step(model, optimizer, scheduler, cfg, x1, y1)
    data_hash = sha(Path(cfg["data"]["shards"]) / "manifest.json")
    tokenizer_hash = sha(cfg["data"]["tokenizer"])
    metadata = {"step": 1, "tokens_seen": 8192, "cursor": 8, "training_seconds": 0.0,
                "wall_time": 0.0, "seed": training["seed"], "config_hash": fingerprint(cfg),
                "data_hash": data_hash, "tokenizer_hash": tokenizer_hash}
    with tempfile.TemporaryDirectory(prefix="prompt3_resume_smoke_") as temp:
        checkpoint = Path(temp) / "state.pt"
        save_training(checkpoint, model, optimizer, scheduler, metadata)
        x2, y2 = PackedStream(Path(cfg["data"]["shards"]), "train", training["context"], 42, 8).next(training["microbatch"], "cuda")
        expected_loss = train_step(model, optimizer, scheduler, cfg, x2, y2)
        expected_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        expected_optimizer = optimizer_snapshot(optimizer)
        expected_scheduler = copy.deepcopy(scheduler.state_dict())
        del model, optimizer, scheduler, stream
        gc_cuda()

        restored = LanguageModel(cfg).cuda()
        restored_optimizer, restored_scheduler = make_optimizer(restored, cfg)
        state = resume_training(checkpoint, restored, restored_optimizer, restored_scheduler, "cuda")
        assert state["cursor"] == 8 and state["tokens_seen"] == 8192
        resumed_loss = train_step(restored, restored_optimizer, restored_scheduler, cfg, x2, y2)
        max_weight_diff = max(float((expected_weights[k] - v.detach().cpu()).abs().max())
                              for k, v in restored.state_dict().items())
        optimizer_diff, optimizer_ok = compare_optimizer(expected_optimizer, optimizer_snapshot(restored_optimizer))
        scheduler_ok = expected_scheduler == restored_scheduler.state_dict()
        passed = abs(expected_loss - resumed_loss) <= 1e-6 and max_weight_diff <= 1e-6 and optimizer_ok and scheduler_ok
        report = {"passed": passed, "model": tag, "seed": training["seed"], "context": training["context"],
                  "microbatch": training["microbatch"], "accumulation": training["accumulation"],
                  "effective_tokens_per_update": 8192, "initial_loss": loss1,
                  "post_resume_reference_loss": expected_loss, "post_resume_restored_loss": resumed_loss,
                  "max_parameter_abs_difference_after_next_update": max_weight_diff,
                  "optimizer_state_max_abs_difference": optimizer_diff,
                  "optimizer_state_equal_within_1e-5": optimizer_ok,
                  "optimizer_resume_atol": OPTIMIZER_RESUME_ATOL,
                  "scheduler_state_equal": scheduler_ok,
                  "scope": "One disposable GPU update, atomic full-state save/restore, then a matched next update. Not quality training."}
        del restored, restored_optimizer, restored_scheduler
        gc_cuda()
        if not passed:
            raise RuntimeError(f"Prompt-3 resume parity failed for {tag}: {report}")
        return report


def gc_cuda() -> None:
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def validate_static_inputs() -> dict:
    from tools.count_params import count_model
    from src.eval.research import freeze_evaluation
    from src.training.research import RESULTS
    manifest_path = Path("results/research_v1/corpus_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["tokens"]["train"] == 210_012_790
    assert manifest["tokens"]["val"] == 8_393_991
    configs = {tag: load_config(f"configs/prompt3/{tag}.yaml") for tag in VARIANTS}
    base = configs[VARIANTS[0]]
    for tag, cfg in configs.items():
        assert cfg["training"] == base["training"], f"training policy mismatch: {tag}"
        assert cfg["data"] == base["data"], f"data policy mismatch: {tag}"
        assert cfg["training"]["context"] == 1024 and cfg["training"]["microbatch"] == 8
        assert cfg["training"]["microbatch"] * cfg["training"]["accumulation"] == 8
        if tag == "dense75_ref":
            assert not cfg["model"]["moe_layers"] and not cfg["memory"]["enabled"]
        else:
            assert cfg["model"]["moe_backend"] == "grouped"
            assert cfg["model"]["moe_layers"] == [3, 7, 11, 15]
            assert cfg["model"]["experts"] == 12 and cfg["model"]["top_k"] == 1
        count = count_model(LanguageModel(cfg))
        assert count["total"] == json.loads(Path("results/prompt3/parameter_counts.json").read_text())[tag]["total"]
    verify_shards(base)
    index = freeze_evaluation(base)
    assert index["context"] == 1024 and index["microbatch"] == 2
    bench = json.loads(Path("results/prompt3/throughput/recommendation.json").read_text(encoding="utf-8"))
    assert bench["microbatch"] == 8 and bench["minimum_reserved_headroom_fraction"] >= 0.15
    return {"manifest_sha256": sha(manifest_path), "tokenizer_sha256": sha(base["data"]["tokenizer"]),
            "evaluation_index_sha256": sha(RESULTS / "evaluation_index.json"), "train_tokens": manifest["tokens"]["train"],
            "validation_tokens": manifest["tokens"]["val"], "variant_count": len(configs),
            "config_fingerprints": {tag: fingerprint(configs[tag]) for tag in VARIANTS}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Pass --run to execute the bounded P3 GPU resume smoke")
    require_cuda("cuda")
    static = validate_static_inputs()
    test_path = ROOT / "tests.xml"
    count, failures = junit_summary(test_path)
    if failures or count < 49:
        raise RuntimeError(f"Prompt-3 full regression receipt missing/failed: tests={count}, failures={failures}")
    smoke = [gpu_resume_smoke(tag) for tag in VARIANTS]
    assert all(row["passed"] for row in smoke)
    report = {"passed": True, "source_hash": source_hash(), "static_inputs": static,
              "junit": "results/prompt3/tests.xml", "test_count": count, "test_failures": failures,
              "gpu_resume_smoke": smoke,
              "authorization_and_scope": "User explicitly authorized Prompt-3 GOOD plan despite Prompt-2 factual MIXED outcome. Bounded preflight only; no long quality training started."}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "gpu_preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    gate = {"passed": True, "source_hash": report["source_hash"], "config_fingerprints": static["config_fingerprints"],
            "data_manifest_sha256": static["manifest_sha256"], "tokenizer_sha256": static["tokenizer_sha256"],
            "evaluation_index_sha256": static["evaluation_index_sha256"], "test_count": count,
            "gpu_resume_smoke_passed": True, "throughput_selected_microbatch": 8,
            "project_write_cap_gib": 45,
            "scope": "Prompt-3 local engineering/data/readiness gate; does not certify quality or automatically start the first stage."}
    (ROOT / "readiness_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "test_count": count, "gpu_variants": len(smoke), "source_hash": gate["source_hash"]}, indent=2))


if __name__ == "__main__":
    main()

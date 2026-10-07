import json
from pathlib import Path

from src.config import fingerprint, load_config


VARIANTS = ("dense75_ref", "sparse75", "sparse75_ngram10m", "sparse75_ngram25m")
GATES = (100_007_936, 250_003_456, 500_006_912, 1_000_005_632)


def test_prompt3_variant_configs_share_frozen_policy_and_data():
    configs = [load_config(f"configs/prompt3/{tag}.yaml") for tag in VARIANTS]
    for cfg in configs:
        assert cfg["model"]["d_model"] == 512
        assert cfg["model"]["layers"] == 16
        assert cfg["model"]["q_heads"] == 8
        assert cfg["model"]["kv_heads"] == 2
        assert cfg["model"]["top_k"] == 1
        assert cfg["training"]["context"] == 1024
        assert cfg["training"]["microbatch"] == 8
        assert cfg["training"]["accumulation"] == 1
        assert cfg["training"]["target_tokens"] == GATES[-1]
        assert cfg["training"]["schedule_steps"] == GATES[-1] // 8192
        assert cfg["data"]["seed"] == 42
        assert cfg["data"]["manifest"] == "results/research_v1/corpus_manifest.json"
    assert all(cfg["training"] == configs[0]["training"] for cfg in configs)
    assert all(cfg["data"] == configs[0]["data"] for cfg in configs)


def test_prompt3_memory_capacities_and_design_receipt_match_configs():
    design = json.loads(Path("results/prompt3/design.json").read_text(encoding="utf-8"))
    rows = {row["variant"]: row for row in design["variants"]}
    expected = {
        "dense75_ref": (54_018_560, 0),
        "sparse75": (76_063_232, 0),
        "sparse75_ngram10m": (86_565_889, 10_485_760),
        "sparse75_ngram25m": (101_245_953, 25_165_824),
    }
    for tag, (stored, table_values) in expected.items():
        cfg = load_config(f"configs/prompt3/{tag}.yaml")
        row = rows[tag]
        assert fingerprint(cfg) == row["config_fingerprint"]
        assert row["total_stored"] == stored
        assert row["memory_tables"] == table_values
        if cfg["memory"]["enabled"]:
            assert cfg["memory"]["banks"] * cfg["memory"]["rows"] * cfg["memory"]["dim"] == table_values


def test_prompt3_stage_gates_are_checkpoint_aligned_and_cumulative():
    assert tuple(sorted(GATES)) == GATES
    assert all(tokens % 8192 == 0 for tokens in GATES)
    assert [tokens // 8192 for tokens in GATES] == [12_208, 30_518, 61_036, 122_071]

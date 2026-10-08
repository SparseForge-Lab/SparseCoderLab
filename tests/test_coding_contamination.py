import hashlib
import json
from pathlib import Path

from tools.check_coding_contamination import fingerprints, inspect_record
from tools.repository_fim import transform


def record(body):
    return {"repository": "example/source", "revision": "a"*40, "file_path": "example.py",
            "text": "<repo>example/source<file>example.py\n"+body,
            "kind": "code", "language": "Python", "raw_sha256": hashlib.sha256(body.encode()).hexdigest()}


def test_prompt_overlap_survives_fim_reordering():
    benchmark = json.loads(Path("configs/eval/simple_python_v1.json").read_text())
    registry = fingerprints(benchmark)
    sample = record(benchmark["tasks"][0]["reference"] + "\n# end\n"*20)
    transformed = transform(sample, seed=42, ratio=1)
    assert transformed["fim"]["applied"]
    assert inspect_record(transformed, registry) == inspect_record(sample, registry)
    assert {"task_id": "add", "kind": "prompt_phrase"} in inspect_record(transformed, registry)


def test_short_generic_snippet_is_not_claimed_as_exact_contamination():
    benchmark = json.loads(Path("configs/eval/simple_python_v1.json").read_text())
    registry = fingerprints(benchmark)
    assert not next(p for p in registry if p["task_id"] == "add")["reference_tokens"]
    assert not inspect_record(record("def add(a, b):\n    return a + b"), registry)


def test_long_exact_reference_hit():
    benchmark = json.loads(Path("configs/eval/simple_python_v1.json").read_text())
    registry = fingerprints(benchmark)
    task = next(t for t in benchmark["tasks"] if t["id"] == "fibonacci")
    hits = inspect_record(record(task["reference"]), registry)
    assert {"task_id": "fibonacci", "kind": "exact_reference_tokens"} in hits

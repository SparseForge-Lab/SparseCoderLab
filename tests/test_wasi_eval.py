"""CPU integration checks for the real Python evaluator and its capabilities."""
import copy
import json
from pathlib import Path

import pytest

from src.eval.wasi_python import WasiPython
from tools.score_saved_python import equivalent, score, validate_rows


@pytest.fixture(scope="module")
def runtime():
    if not Path(".local/wasi_eval/integrity.json").exists():
        pytest.skip("Install pinned CPU runtime with python -m tools.setup_wasi_eval")
    return WasiPython()


def test_host_oracle_types_and_tolerances():
    assert equivalent(1.0, 1)
    assert equivalent(68.00000001, 68)
    assert not equivalent(1, True)
    assert not equivalent(True, 1)
    assert not equivalent(None, 0)
    assert equivalent([1, None, False], [1, None, False])
    assert not equivalent([1, None, 0], [1, None, False])
    assert not equivalent(10**4000, 1)


def test_frozen_tasks_reference_solutions(runtime):
    benchmark = json.loads(Path("configs/eval/simple_python_v1.json").read_text())
    assert len(benchmark["tasks"]) == 25
    assert sum(len(t["cases"]) for t in benchmark["tasks"]) == 94
    for task in benchmark["tasks"]:
        result = score(runtime, task["reference"], task)
        assert result["outcome"] == "correct", (task["id"], result)
        assert result["loop_constraint_met"]


@pytest.mark.parametrize("body", [
    'return open("/host-project/README.md").read()',
    'return open("/runtime/lib/../../README.md").read()',
    'return open("/runtime/lib/forbidden_write", "w").write("x")',
    'import socket\n    return socket.create_connection(("127.0.0.1",8765))',
    'import os\n    return os.system("echo forbidden")',
])
def test_guest_cannot_access_host_files_write_or_network(runtime, body):
    result = runtime.execute("def probe():\n    " + body, "probe", [[]])
    assert result["kind"] == "runtime_error"
    assert not Path(".local/wasi_eval/cpython/lib/forbidden_write").exists()


def test_guest_does_not_inherit_host_environment(runtime, monkeypatch):
    monkeypatch.setenv("PRIVATE_EVAL_TEST_VALUE", "not-visible")
    result = runtime.execute('def probe():\n    import os\n    return os.getenv("PRIVATE_EVAL_TEST_VALUE")', "probe", [[]])
    assert result["kind"] == "completed"
    assert result["results"] == [{"value": None, "stdout": ""}]


def test_limits_and_fresh_store_recovery(runtime):
    source = "def probe():\n    return 7\n" + "# padding\n" * 2000
    task = dict(id='probe',prompt='def probe():',function='probe',cases=[dict(args=[],expected=7,stdout='')])
    limited = score(runtime, source, task)
    assert limited['outcome'] == 'source_limit' and limited['execution_seconds'] == 0 and limited['fuel_consumed'] == 0
    result = runtime.execute("def probe():\n    while True: pass", "probe", [[]])
    assert result["kind"] == "timeout"
    assert result["wall_seconds"] < 6
    result = runtime.execute('def probe():\n    return bytearray(1024**3)', "probe", [[]])
    assert result["kind"] == "runtime_error"
    result = runtime.execute('def probe():\n    import os\n    os.write(1, b"x" * 20000)', "probe", [[]])
    assert result["kind"] == "output_limit"
    assert runtime.execute("def probe():\n    return 7", "probe", [[]])["results"][0]["value"] == 7


def test_host_rejects_fake_success_and_keeps_repetition_separate(runtime):
    task = json.loads(Path("configs/eval/simple_python_v1.json").read_text())["tasks"][0]
    wrong = score(runtime, task["prompt"]+'\n    print("PASS")\n    return 999', task)
    assert wrong["outcome"] == "wrong"
    repeated = score(runtime, task["prompt"]+'\n    return a+b\n    return a+b\n    return a+b\n    return a+b', task)
    assert repeated["functional_correct"] and repeated["repetition"]


def test_saved_study_comparison_rejects_missing_duplicate_or_mismatched_rows():
    source = Path("results/prompt3_cpu_generation_benchmark.jsonl")
    if not source.exists():
        pytest.skip("Historical generation payload is retained locally")
    rows = [json.loads(s) for s in source.open(encoding="utf8")]
    tasks = json.loads(Path("configs/eval/simple_python_v1.json").read_text())["tasks"]
    manifest = json.loads(Path("results/prompt3/transition_checkpoint_manifest.json").read_text())
    policy = validate_rows(rows, tasks, manifest)
    assert policy["0.0"]["top_k"] is None and policy["0.2"]["top_k"] == 40
    with pytest.raises(ValueError):
        validate_rows(rows[:-1], tasks, manifest)
    with pytest.raises(ValueError):
        validate_rows(rows + [rows[0]], tasks, manifest)
    altered = copy.deepcopy(rows)
    altered[0]["training_tokens"] += 8192
    with pytest.raises(ValueError):
        validate_rows(altered, tasks, manifest)
    altered = copy.deepcopy(rows)
    altered[0]["top_k"] = 40
    with pytest.raises(ValueError):
        validate_rows(altered, tasks, manifest)
    attested = copy.deepcopy(rows)
    for row in attested:
        row['checkpoint_sha256'] = manifest['canonical_transition_checkpoints'][row['variant']]['model_weights']['sha256']
    assert validate_rows(attested,tasks,manifest)==policy
    attested[0]['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError,match='checkpoint hash'):validate_rows(attested,tasks,manifest)

from src.eval.target_function import extract_target_function, score_target


def task():
    return {"id": "sum", "function": "add", "cases": [
        {"args": [2, 3], "expected": 5, "stdout": ""}]}


class Runtime:
    def execute(self, source, function, arguments):
        if "raise RuntimeError" in source:
            return {"kind": "runtime_error", "exception": "RuntimeError"}
        value = 999 if "return 999" in source else sum(arguments[0])
        return {"kind": "completed", "results": [{"value": value, "stdout": ""}]}


class TimeoutRuntime:
    def execute(self, source, function, arguments):
        return {"kind": "timeout", "wall_seconds": 2.0, "fuel_consumed": 99}


def test_extracts_complete_body_and_ignores_malformed_trailing_text():
    prompt = "def add(a, b):"
    result = score_target("\n    return a + b\nthis is ( broken", prompt, task(), Runtime())
    assert result["target_outcome"] == "pass"
    assert result["target_parseable"]
    assert result["syntax_damage_after_target"]


def test_first_complete_requested_function_is_primary_and_counts_redefinition():
    text = "def add(a, b):\n    return a + b\n\ndef add(a, b):\n    return 999\n"
    result = score_target(text, "", task(), Runtime())
    assert result["target_definitions"] == 2
    assert result["target_outcome"] == "pass"
    assert result["last_definition_outcome"] == "wrong"
    assert result["later_redefinition_overwrites_earlier_correct"]


def test_keeps_only_needed_imports_and_simple_helpers():
    prompt = ""
    text = "import math\ndef helper(x):\n    return math.floor(x)\n\ndef add(a, b):\n    return helper(a + b)\n"
    extracted = extract_target_function(text, prompt, task())
    assert extracted["target_status"] == "complete"
    assert "import math" in extracted["target_code"]
    assert "def helper" in extracted["target_code"]


def test_classifies_extraction_failure_and_incomplete_target():
    assert extract_target_function("explanation only", "", task())["target_status"] == "extraction_failure"
    result = extract_target_function("def add(a, b):\n    return (", "", task())
    assert result["target_status"] == "syntax_error"


def test_body_fence_and_target_end_metadata():
    result = extract_target_function("```python\n    return a + b\n```\n", "def add(a, b):", task())
    assert result["target_status"] == "complete"
    assert result["extraction"] == "fenced"
    assert result["target_end_char"] is not None


def test_target_outcome_categories_include_wrong_runtime_and_timeout():
    wrong = score_target("def add(a, b):\n    return 999\n", "", task(), Runtime())
    assert wrong["target_outcome"] == "wrong" and wrong["target_parseable"]
    runtime_error = score_target("def add(a, b):\n    raise RuntimeError()\n", "", task(), Runtime())
    assert runtime_error["target_outcome"] == "runtime_error"
    timeout = score_target("def add(a, b):\n    return a + b\n", "", task(), TimeoutRuntime())
    assert timeout["target_outcome"] == "timeout"


def test_extracts_only_referenced_safe_prelude():
    code = ("import math\nimport json\ndef helper(x):\n    return math.floor(x)\n"
            "unused = 9\ndef add(a, b):\n    return helper(a + b)\n")
    result = extract_target_function(code, "", task())
    assert "import math" in result["target_code"]
    assert "def helper" in result["target_code"]
    assert "import json" not in result["target_code"]
    assert "unused" not in result["target_code"]

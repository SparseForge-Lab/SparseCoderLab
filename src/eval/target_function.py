"""AST-based evaluation of the requested function inside saved completions.

This scorer intentionally keeps the full-completion score separate. It takes
the first syntactically complete requested function from the emitted text and
executes that extracted module only through the existing WASI runtime.
"""
from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter


def _fenced_or_raw(text: str) -> tuple[str | None, str]:
    blocks = re.findall(r"```(?:[A-Za-z0-9_+.-]+)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return blocks[0].strip("\r\n"), "fenced"
    code = text.strip("\r\n")
    if re.search(r"(?m)^\s*(?:async\s+def\b|def\b|class\b|from\b|import\b|return\b|if\b|for\b|while\b|try\b|with\b|raise\b|[A-Za-z_]\w*\s*=)", code):
        return code, "raw"
    return None, "no_code"


def _function_nodes(tree: ast.Module, name: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name]


def _parse_prefixes(source: str, start_line: int, function_name: str) -> tuple[list[ast.Module], str | None]:
    """Parse complete line prefixes, retaining the last valid module snapshot.

    No syntax is repaired. Parsing at line boundaries permits a complete target
    definition to survive malformed or unrelated text emitted afterwards.
    """
    lines = source.splitlines(keepends=True)
    snapshots: list[ast.Module] = []
    last_error = None
    for end in range(1, len(lines) + 1):
        prefix = "".join(lines[:end])
        try:
            tree = ast.parse(prefix)
        except (SyntaxError, ValueError) as error:
            last_error = type(error).__name__
            continue
        nodes = [node for node in _function_nodes(tree, function_name) if node.lineno >= start_line]
        if nodes:
            snapshots.append(tree)
    return snapshots, last_error


def _bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {alias.asname or alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        out = set()
        for target in targets:
            out.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        return out
    return set()


def _literal_assignment(node: ast.stmt) -> bool:
    value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
    if value is None:
        return False
    try:
        ast.literal_eval(value)
        return True
    except (ValueError, TypeError):
        return False


def _needed_prelude(module: ast.Module, target: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt] | None:
    """Select prior imports/helpers referenced by the target, conservatively."""
    prior = [node for node in module.body if node is not target and node.end_lineno < target.lineno]
    candidates: dict[str, ast.stmt] = {}
    for node in prior:
        names = _bound_names(node)
        if not names:
            continue
        # Imports, function helpers and literal constants are safe to retain;
        # class bodies and executable assignments are not inferred as helpers.
        safe = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef))
        safe = safe or _literal_assignment(node)
        if safe:
            for name in names:
                candidates[name] = node
    needed: set[str] = set()
    pending = {n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    selected: list[ast.stmt] = []
    seen_nodes: set[int] = set()
    while pending:
        name = pending.pop()
        node = candidates.get(name)
        if node is None or id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node)); needed.add(name); selected.append(node)
        pending.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    selected.sort(key=lambda n: n.lineno)
    return selected


def _module_for(module: ast.Module, target: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    prelude = _needed_prelude(module, target)
    if prelude is None:
        return None
    # ast.unparse serializes only already-parsed nodes; it does not complete or
    # repair model output. The requested function itself is kept unchanged.
    return "\n".join(ast.unparse(node) for node in [*prelude, target]) + "\n"


def _parse_target(code: str, prompt: str, name: str):
    prompt_lines = prompt.splitlines()
    prompt_has_header = any(re.match(rf"\s*(?:async\s+)?def\s+{re.escape(name)}\s*\(", line)
                            for line in prompt_lines)
    complete_header = bool(re.search(rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(name)}\s*\(", code))
    if complete_header or (code.lstrip().startswith(("import ", "from ")) and "\ndef " in code):
        source, start = code, len(prompt_lines) + 1
    elif prompt_has_header:
        source = prompt.rstrip("\n") + "\n" + (code if code[:1].isspace() else "    " + code)
        start = 1
    else:
        source, start = code, 1
    snapshots, last_error = _parse_prefixes(source, start, name)
    if not snapshots:
        return {"status": "syntax_error" if last_error else "extraction_failure",
                "source": source, "definitions": [], "last_error": last_error}
    # Each snapshot may repeat prior definitions. Keep unique definitions by
    # source location; the first syntactically complete target is authoritative.
    definitions: dict[tuple[int, int], tuple[ast.Module, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for tree in snapshots:
        for node in _function_nodes(tree, name):
            if node.lineno >= start:
                definitions[(node.lineno, node.col_offset)] = (tree, node)
    ordered = [definitions[k] for k in sorted(definitions)]
    return {"status": "complete", "source": source, "definitions": ordered,
            "last_error": last_error}


def extract_target_function(text: str, prompt: str, task: dict) -> dict:
    code, extraction = _fenced_or_raw(text)
    result = {"extraction": extraction, "target_status": "extraction_failure",
              "target_code": None, "target_definitions": 0, "target_end_char": None,
              "target_tokens_after": None, "candidate_sha256": hashlib.sha256(text.encode()).hexdigest()}
    if code is None:
        return result
    parsed = _parse_target(code, prompt, task["function"])
    result["target_status"] = parsed["status"]
    if parsed["status"] != "complete":
        return result
    module, target = parsed["definitions"][0]
    extracted = _module_for(module, target)
    if extracted is None:
        result["target_status"] = "extraction_failure"
        return result
    result["target_code"] = extracted
    result["target_definitions"] = len(parsed["definitions"])
    target_line_end = target.end_lineno
    source_lines = parsed["source"].splitlines(keepends=True)
    target_end_source = sum(map(len, source_lines[:target_line_end]))
    if parsed["source"].startswith(prompt.rstrip("\n")):
        local_end = max(0, target_end_source - len(prompt.rstrip("\n")) - 1)
    else:
        local_end = target_end_source
    result["target_end_char"] = text.find(code) + local_end if text.find(code) >= 0 else None
    result["_definitions"] = parsed["definitions"]
    result["_complete_source"] = parsed["source"]
    result["_target_end_line"] = target.end_lineno
    return result


def score_target(text: str, prompt: str, task: dict, runtime, *, tokenizer=None) -> dict:
    row = extract_target_function(text, prompt, task)
    result = {**row, "target_parseable": row["target_code"] is not None,
              "target_passed_cases": 0, "target_case_count": len(task["cases"]),
              "target_functional_correct": False, "target_outcome": row["target_status"]}
    code = row["target_code"]
    if code is not None:
        execution = runtime.execute(code, task["function"], [case["args"] for case in task["cases"]])
        kind = execution["kind"]
        from tools.score_saved_python import equivalent
        if kind == "completed":
            obs = execution.get("results", [])
            passed = sum(isinstance(o, dict) and equivalent(o.get("value"), case["expected"])
                         and o.get("stdout") == case.get("stdout", "")
                         for o, case in zip(obs, task["cases"]))
            result["target_passed_cases"] = passed
            result["target_observations"] = obs
            outcome = "pass" if len(obs) == len(task["cases"]) and passed == len(task["cases"]) else "wrong"
        elif kind == "timeout":
            outcome = "timeout"
        elif kind == "syntax_error":
            outcome = "syntax_error"
        else:
            outcome = "runtime_error"
        result.update(target_outcome=outcome, target_functional_correct=outcome == "pass",
                      target_execution_seconds=execution.get("wall_seconds"),
                      target_fuel_consumed=execution.get("fuel_consumed"),
                      target_exception=execution.get("exception"))
        end = row.get("target_end_char")
        if tokenizer is not None and end is not None:
            result["target_tokens_after"] = len(tokenizer.encode(text[end:]).ids)
            result["target_tokens_after_method"] = "re-encoded decoded-text suffix; approximate"
        definitions = row.pop("_definitions", [])
        complete_source = row.pop("_complete_source", "")
        target_end_line = row.pop("_target_end_line", 0)
        result.pop("_definitions", None)
        source_lines = complete_source.splitlines()
        tail_lines = source_lines[target_end_line:]
        result["trailing_nonblank_lines"] = sum(bool(line.strip()) for line in tail_lines)
        try:
            ast.parse(complete_source)
            result["full_completion_syntax_valid"] = True
        except (SyntaxError, ValueError):
            result["full_completion_syntax_valid"] = False
        result["syntax_damage_after_target"] = bool(result["trailing_nonblank_lines"] and
                                                      not result["full_completion_syntax_valid"])
        if len(definitions) > 1:
            last_module, last_target = definitions[-1]
            last_code = _module_for(last_module, last_target)
            last_run = runtime.execute(last_code, task["function"], [case["args"] for case in task["cases"]])
            if last_run["kind"] == "completed":
                from tools.score_saved_python import equivalent
                passed = sum(isinstance(o, dict) and equivalent(o.get("value"), case["expected"])
                             and o.get("stdout") == case.get("stdout", "")
                             for o, case in zip(last_run.get("results", []), task["cases"]))
                last_pass = len(last_run.get("results", [])) == len(task["cases"]) and passed == len(task["cases"])
                last_outcome = "pass" if last_pass else "wrong"
            elif last_run["kind"] == "timeout":
                last_pass, last_outcome = False, "timeout"
            elif last_run["kind"] == "syntax_error":
                last_pass, last_outcome = False, "syntax_error"
            else:
                last_pass, last_outcome = False, "runtime_error"
            result.update(last_definition_outcome=last_outcome,
                          last_definition_functional_correct=last_pass,
                          later_redefinition_overwrites_earlier_correct=(
                              result["target_functional_correct"] and not last_pass),
                          first_last_definition_differ=(ast.dump(definitions[0][1]) != ast.dump(last_target)))
        else:
            result.update(last_definition_outcome=result["target_outcome"],
                          last_definition_functional_correct=result["target_functional_correct"],
                          later_redefinition_overwrites_earlier_correct=False,
                          first_last_definition_differ=False)
    else:
        result.pop("_definitions", None)
        result.pop("_complete_source", None)
        result.pop("_target_end_line", None)
        result.pop("_definitions", None)
    return result


def outcome_counts(rows: list[dict]) -> dict:
    counts = Counter(row.get("target_outcome", "unscored") for row in rows)
    return {"outcomes": dict(sorted(counts.items())),
            "parseable_target_count": sum(bool(row.get("target_parseable")) for row in rows),
            "target_pass_at_1": sum(bool(row.get("target_functional_correct")) for row in rows) / len(rows) if rows else None}

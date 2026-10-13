"""Contracts for paired STOP/CONTINUE examples and output-scope measurements."""

from __future__ import annotations

from typing import Any, Mapping


def validate_stop_continue_pair(stop: Mapping[str, Any], continuation: Mapping[str, Any]) -> None:
    if not stop.get("pair_id") or stop.get("pair_id") != continuation.get("pair_id"):
        raise ValueError("STOP and CONTINUE records must share a pair_id")
    if stop.get("stop_should_happen") is not True:
        raise ValueError("STOP member must set stop_should_happen=true")
    if continuation.get("stop_should_happen") is not False:
        raise ValueError("CONTINUE member must set stop_should_happen=false")
    if stop.get("requested_scope") == continuation.get("requested_scope"):
        raise ValueError("paired records need distinct requested scopes")
    for row in (stop, continuation):
        if not row.get("sample_id") or not row.get("task_family"):
            raise ValueError("paired records require sample_id and task_family metadata")


def stop_scope_metrics(*, target_complete: bool, natural_stop: bool, cap_reached: bool,
                       tokens_after_target: int, repetitions_after_target: int,
                       continuation_required: bool) -> dict[str, Any]:
    if tokens_after_target < 0 or repetitions_after_target < 0:
        raise ValueError("continuation counts cannot be negative")
    return {"target_complete": bool(target_complete), "natural_stop": bool(natural_stop),
            "cap_reached": bool(cap_reached), "tokens_after_target": tokens_after_target,
            "repetition_after_target": repetitions_after_target,
            "continuation_required": bool(continuation_required),
            "premature_stop": bool(continuation_required and natural_stop and not target_complete)}

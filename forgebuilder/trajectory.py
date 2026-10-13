"""Compact evidence/action trajectories with explicit target masking."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .schema import canonical_json, sha256_bytes

EVENT_TYPES = {"EVIDENCE", "PLAN", "ACTION", "OBSERVATION", "DIAGNOSIS", "VERIFY", "FINAL",
               "WRONG_HYPOTHESIS", "FAILURE_OUTPUT", "RECOVERY_ACTION", "TOOL_CALL", "TOOL_RESULT"}


def normalize_trajectory(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ValueError("trajectory cannot be empty")
    normalized = []
    for index, event in enumerate(events):
        kind = event.get("type")
        if kind not in EVENT_TYPES:
            raise ValueError(f"unknown trajectory event type at index {index}: {kind!r}")
        content = event.get("content")
        if not isinstance(content, str):
            raise ValueError("trajectory event content must be text")
        row = {"sequence": index, "type": kind, "content": content,
               "evidence_refs": list(event.get("evidence_refs", [])),
               "tool_name": event.get("tool_name"), "tool_args": event.get("tool_args"),
               "tool_result_ref": event.get("tool_result_ref"),
               "known_bad": bool(event.get("known_bad", False))}
        masked = row["known_bad"] or kind in {"WRONG_HYPOTHESIS", "FAILURE_OUTPUT"}
        row["loss_mask"] = 0.0 if masked else float(event.get("loss_mask", 1.0))
        row["target_weight"] = 0.0 if masked else float(event.get("target_weight", 1.0))
        if row["loss_mask"] < 0 or row["target_weight"] < 0:
            raise ValueError("loss mask and target weight must be non-negative")
        normalized.append(row)
    if normalized[-1]["type"] != "FINAL":
        raise ValueError("trajectory must end with FINAL")
    payload = {"schema_version": 1, "events": normalized}
    payload["tool_trace_hash"] = sha256_bytes(canonical_json(normalized))
    return payload

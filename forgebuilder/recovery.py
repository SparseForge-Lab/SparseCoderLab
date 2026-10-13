"""Student mistake and verified recovery records for future controlled learning."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .pipeline import VerificationResult
from .schema import canonical_json, sha256_bytes

FAILURE_TYPES = {"invented_api", "wrong_symbol", "wrong_file", "did_not_inspect_repo", "false_assumption",
                 "misread_test", "loop_after_correct", "continued_after_complete", "stopped_too_early",
                 "failed_recovery", "incorrect_tool_use", "ignored_evidence", "other"}


def validate_student_failure(record: Mapping[str, Any]) -> dict[str, Any]:
    required = ("sample_id", "student_model", "student_version", "failure_type", "state", "evidence",
                "wrong_hypothesis", "environment")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError("missing student-failure fields: " + ", ".join(missing))
    if record["failure_type"] not in FAILURE_TYPES:
        raise ValueError("unrecognized failure_type")
    row = dict(record)
    row.setdefault("recurrence_count", 1)
    row.setdefault("confidence_before_failure", None)
    row.setdefault("similarity_reference", None)
    row["known_bad_actions"] = [dict(action, loss_mask=0.0, target_weight=0.0)
                                for action in row.get("known_bad_actions", [])]
    if not isinstance(row["recurrence_count"], int) or row["recurrence_count"] < 1:
        raise ValueError("recurrence_count must be positive")
    confidence = row["confidence_before_failure"]
    if confidence is not None and not 0 <= confidence <= 1:
        raise ValueError("confidence_before_failure must be between zero and one")
    canonical_json(row)
    return row


def validate_recovery(*, parent_sample_id: str, failure: Mapping[str, Any], diagnosis: str,
                      recovery_actions: Sequence[Mapping[str, Any]], verification: VerificationResult,
                      final_result: str, recovery_source: str) -> dict[str, Any]:
    if not parent_sample_id or not diagnosis.strip() or not recovery_actions or not final_result.strip():
        raise ValueError("recovery must link a parent and include diagnosis, actions, and final result")
    normalized_failure = validate_student_failure(failure)
    if normalized_failure["sample_id"] != parent_sample_id:
        raise ValueError("parent_sample_id does not match the linked failure")
    verified = bool(verification.status == "PASS" and verification.passed > 0
                    and verification.test_hash and verification.environment_hash)
    row = {"schema_version": 1, "sample_id": "recovery-" + sha256_bytes(canonical_json({
        "parent": parent_sample_id, "diagnosis": diagnosis, "actions": recovery_actions,
        "test_hash": verification.test_hash, "environment_hash": verification.environment_hash})),
        "parent_sample_id": parent_sample_id, "failure_type": normalized_failure["failure_type"],
        "wrong_hypothesis": normalized_failure["wrong_hypothesis"], "evidence": normalized_failure["evidence"],
        "diagnosis": diagnosis, "recovery_actions": list(recovery_actions),
        "verification": verification.__dict__, "recovery_verified": verified,
        "accepted": verified, "recovery_source": recovery_source,
        "student_model": normalized_failure["student_model"],
        "student_version": normalized_failure["student_version"], "final_result": final_result,
        "recurrence_count": normalized_failure["recurrence_count"],
        "confidence_before_failure": normalized_failure["confidence_before_failure"],
        "similarity_reference": normalized_failure["similarity_reference"]}
    if not recovery_source:
        raise ValueError("recovery_source must identify the source of the recovery")
    return row


def failure_priority(record: Mapping[str, Any]) -> float:
    row = validate_student_failure(record)
    high_value = {"invented_api", "false_assumption", "misread_test", "loop_after_correct",
                  "continued_after_complete", "stopped_too_early", "failed_recovery", "ignored_evidence"}
    score = 1.0 + min(row["recurrence_count"] - 1, 10) * 0.5
    if row["failure_type"] in high_value:
        score += 2.0
    if row.get("objectively_verified"):
        score += 2.0
    if row.get("high_confidence_but_wrong"):
        score += 1.0
    return score

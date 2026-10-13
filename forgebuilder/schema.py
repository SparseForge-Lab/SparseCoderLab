"""Versioned records used by foundation, post-training, and recovery data."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re
from typing import Any, Mapping

SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = {"train", "validation", "test", "quarantine", "unassigned"}
_REALITY = {"real", "generated", "derived", "student_on_policy"}
_DATA_STAGES = {"foundation", "posttraining", "student_recovery"}


class SchemaError(ValueError):
    """Raised when a normalized sample is incomplete or internally invalid."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_sample_id(record: Mapping[str, Any]) -> str:
    """Return an ID stable across metadata enrichment and serialization changes."""
    identity = {
        "schema_version": int(record.get("schema_version", SCHEMA_VERSION)),
        "source_id": record.get("source_id"),
        "source_type": record.get("source_type"),
        "content_sha256": record.get("content_sha256"),
        "task_family": record.get("task_family"),
        "transformation_recipe": record.get("transformation_recipe"),
        "parent_sample_ids": sorted(record.get("parent_sample_ids") or []),
    }
    if not identity["source_id"] or not identity["source_type"] or not identity["content_sha256"]:
        raise SchemaError("source_id, source_type, and content_sha256 are needed for a stable ID")
    return "ds2-" + sha256_bytes(canonical_json(identity))


def validate_record(record: Mapping[str, Any], *, assign_id: bool = True) -> dict[str, Any]:
    """Validate a JSON-compatible Dataset-v2 record and return a normalized copy.

    Unknown fields are retained so later schema versions can be read without
    dropping modality- or source-specific metadata.
    """
    if not isinstance(record, Mapping):
        raise SchemaError("record must be a mapping")
    normalized = dict(record)
    version = normalized.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise SchemaError(f"unsupported schema_version: {version!r}")
    normalized["schema_version"] = SCHEMA_VERSION
    required = ("source_id", "source_type", "content_sha256", "task_family", "real_or_generated", "data_stage")
    missing = [key for key in required if not isinstance(normalized.get(key), str) or not normalized[key].strip()]
    if missing:
        raise SchemaError("missing required non-empty fields: " + ", ".join(missing))
    if not _SHA256.fullmatch(normalized["content_sha256"]):
        raise SchemaError("content_sha256 must be a lowercase SHA-256 hex digest")
    if normalized["real_or_generated"] not in _REALITY:
        raise SchemaError(f"invalid real_or_generated: {normalized['real_or_generated']!r}")
    if normalized["data_stage"] not in _DATA_STAGES:
        raise SchemaError(f"invalid data_stage: {normalized['data_stage']!r}")
    split = normalized.get("split", "unassigned")
    if split not in _SPLITS:
        raise SchemaError(f"invalid split: {split!r}")
    normalized["split"] = split
    if "license" not in normalized:
        raise SchemaError("license must be present; use null when unknown")
    if normalized["license"] is not None and (not isinstance(normalized["license"], str) or not normalized["license"].strip()):
        raise SchemaError("license must be a non-empty identifier or null")
    if "source_provenance" not in normalized:
        raise SchemaError("source_provenance must be present; do not invent provenance")
    if normalized["source_provenance"] is None:
        raise SchemaError("source_provenance must describe known provenance")
    if not isinstance(normalized["source_provenance"], (dict, list, str)):
        raise SchemaError("source_provenance must be JSON-compatible provenance data")
    if not normalized["source_provenance"] or (isinstance(normalized["source_provenance"], str)
                                                 and not normalized["source_provenance"].strip()):
        raise SchemaError("source_provenance cannot be empty")
    for key in ("execution_verified", "compiler_verified", "test_verified", "stop_should_happen",
                "recovery_verified", "accepted"):
        if key in normalized and normalized[key] is not None and not isinstance(normalized[key], bool):
            raise SchemaError(f"{key} must be boolean or null")
    for key in ("test_hash", "environment_hash", "tool_trace_hash"):
        value = normalized.get(key)
        if value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value)):
            raise SchemaError(f"{key} must be null or a lowercase SHA-256 digest")
    if "token_count" in normalized and normalized["token_count"] is not None:
        if isinstance(normalized["token_count"], bool) or not isinstance(normalized["token_count"], int) or normalized["token_count"] < 0:
            raise SchemaError("token_count must be a non-negative integer or null")
    for key in ("parent_sample_ids", "natural_language"):
        value = normalized.get(key)
        if value is not None and not isinstance(value, list):
            raise SchemaError(f"{key} must be a list or null")
    parents = normalized.get("parent_sample_ids") or []
    if any(not isinstance(item, str) or not item for item in parents):
        raise SchemaError("parent_sample_ids contains an invalid ID")
    created = normalized.get("created_at")
    if created is not None:
        if not isinstance(created, str):
            raise SchemaError("created_at must be an ISO-8601 string or null")
        try:
            datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaError("created_at is not valid ISO-8601") from exc
    if normalized.get("accepted") is False and not normalized.get("rejection_reason"):
        raise SchemaError("rejected records require a rejection_reason")
    if assign_id and not normalized.get("sample_id"):
        normalized["sample_id"] = stable_sample_id(normalized)
    elif normalized.get("sample_id") and not isinstance(normalized["sample_id"], str):
        raise SchemaError("sample_id must be a string")
    try:
        canonical_json(normalized)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"record must be finite JSON data: {exc}") from exc
    return normalized

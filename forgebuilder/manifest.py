"""Deterministic manifests, repository-level splits, and eval quarantine."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

from .schema import canonical_json, sha256_bytes, stable_sample_id, validate_record


def assign_repository_split(repo_id: str, *, seed: int = 42,
                            ratios: Mapping[str, float] | None = None) -> str:
    """Assign every record from a repository to the same deterministic split."""
    ratios = dict(ratios or {"train": 0.90, "validation": 0.05, "test": 0.05})
    if not repo_id or not ratios or any(v < 0 for v in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("repo_id is required and split ratios must be non-negative and sum to one")
    bucket = int.from_bytes(hashlib.sha256(f"{seed}\0{repo_id}".encode()).digest()[:8], "big") / 2**64
    edge = 0.0
    for split, ratio in ratios.items():
        edge += ratio
        if bucket < edge:
            return split
    return next(reversed(ratios))


def write_manifest(path: Path, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Atomically write a canonical JSONL manifest ordered by stable sample ID."""
    rows = [validate_record(row) for row in records]
    rows.sort(key=lambda row: row["sample_id"])
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate sample_id values")
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return {"schema_version": 1, "record_count": len(rows), "sha256": sha256_bytes(payload),
            "sample_ids": ids, "path": path.name}


def read_manifest(path: Path, *, verify_hash: str | None = None) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    if verify_hash and sha256_bytes(raw) != verify_hash:
        raise ValueError("manifest SHA-256 does not match")
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            rows.append(validate_record(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid manifest row {line_number}: {exc}") from exc
    if [r["sample_id"] for r in rows] != sorted(r["sample_id"] for r in rows):
        raise ValueError("manifest rows are not in stable sample_id order")
    return rows


def _lexical_shingles(text: str, width: int = 8) -> set[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.casefold())
    if not tokens:
        return set()
    shingles = (" ".join(tokens[i:i + width]) for i in range(max(1, len(tokens) - width + 1)))
    return {hashlib.sha256(s.encode()).hexdigest() for s in shingles}


class EvalQuarantine:
    """A lexical contamination screen; a clear result is not a universal guarantee."""

    def __init__(self, benchmarks: Iterable[Mapping[str, Any]], *, shingle_width: int = 8):
        if shingle_width < 2:
            raise ValueError("shingle_width must be at least two")
        self.shingle_width = shingle_width
        self._registry: list[tuple[str, str, set[str]]] = []
        for benchmark in benchmarks:
            benchmark_id = str(benchmark["benchmark_id"])
            for field in ("prompt", "answer", "reference"):
                value = benchmark.get(field)
                if isinstance(value, str) and value.strip():
                    self._registry.append((benchmark_id, field, _lexical_shingles(value, shingle_width)))

    def inspect(self, text: str, *, threshold: float = 0.80) -> dict[str, Any]:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        sample = _lexical_shingles(text, self.shingle_width)
        matches = []
        for benchmark_id, field, expected in self._registry:
            if not expected:
                continue
            overlap = len(sample & expected) / len(expected)
            if overlap >= threshold:
                matches.append({"benchmark_id": benchmark_id, "field": field,
                                "reference_shingle_overlap": round(overlap, 6)})
        return {"status": "flagged" if matches else "clear_lexical_screen",
                "matches": matches, "method": "token_shingle_overlap", "threshold": threshold,
                "universal_contamination_claim": False}

    def annotate(self, record: Mapping[str, Any], text: str) -> dict[str, Any]:
        result = dict(record)
        screen = self.inspect(text)
        result["eval_quarantine_status"] = screen["status"]
        result["eval_quarantine_evidence"] = screen
        return result

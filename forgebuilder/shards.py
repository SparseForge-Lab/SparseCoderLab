"""Resumable sequential JSONL shards for normalized sample records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator, Mapping

from .schema import canonical_json, validate_record


class ResumableShardWriter:
    def __init__(self, root: Path, *, prefix: str = "records", max_records: int = 100_000):
        if max_records < 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
            raise ValueError("max_records must be positive and prefix must be filename-safe")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.prefix, self.max_records = prefix, max_records
        partials = sorted(self.root.glob(f"{prefix}-*.jsonl.partial"))
        if len(partials) > 1:
            raise ValueError("multiple partial shards found; inspect before resuming")
        finals = sorted(self.root.glob(f"{prefix}-[0-9][0-9][0-9][0-9][0-9].jsonl"))
        self._index = max((int(p.stem.split("-")[-1]) for p in finals), default=-1) + 1
        self._partial = partials[0] if partials else self.root / f"{prefix}-{self._index:05d}.jsonl.partial"
        if partials:
            found = int(self._partial.name.split("-")[-1].split(".")[0])
            if finals and found <= max(int(p.stem.split("-")[-1]) for p in finals):
                raise ValueError("partial shard index conflicts with a finalized shard")
            self._index = found
        self._count = self._validate_and_count(self._partial) if self._partial.exists() else 0
        self._reconcile_manifest(finals)
        if self._count > self.max_records:
            raise ValueError("partial shard already exceeds max_records")
        self._stream = self._partial.open("ab")

    def _validate_and_count(self, path: Path) -> int:
        count = 0
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = validate_record(json.loads(line))
                    if row.get("sample_id") != json.loads(line).get("sample_id"):
                        raise ValueError("stored sample ID is missing")
                except Exception as exc:
                    raise ValueError(f"invalid partial shard at line {line_number}: {exc}") from exc
                count += 1
        return count

    def _reconcile_manifest(self, finals: list[Path]) -> None:
        """Recover a finalized shard if a process stopped before manifest replace."""
        manifest_path = self.root / f"{self.prefix}-manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {"schema_version": 1, "shards": []}
        known = {row["path"] for row in manifest["shards"]}
        for path in finals:
            if path.name in known:
                continue
            count = self._validate_and_count(path)
            manifest["shards"].append({"path": path.name, "records": count,
                                        "sha256": _sha256_file(path), "size_bytes": path.stat().st_size})
        for row in manifest["shards"]:
            if not (self.root / row["path"]).is_file():
                raise FileNotFoundError(f"manifest references missing shard: {row['path']}")
        manifest["shards"].sort(key=lambda row: row["path"])
        _atomic_write(manifest_path, canonical_json(manifest) + b"\n")

    def write(self, record: Mapping[str, Any]) -> str:
        row = validate_record(record)
        if self._count >= self.max_records:
            self._finalize()
        self._stream.write(canonical_json(row) + b"\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._count += 1
        return row["sample_id"]

    def _finalize(self) -> None:
        if self._count == 0:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        final = self._partial.with_suffix("")
        os.replace(self._partial, final)
        digest = _sha256_file(final)
        manifest_path = self.root / f"{self.prefix}-manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {"schema_version": 1, "shards": []}
        manifest["shards"] = [r for r in manifest["shards"] if r["path"] != final.name]
        manifest["shards"].append({"path": final.name, "records": self._count, "sha256": digest,
                                    "size_bytes": final.stat().st_size})
        manifest["shards"].sort(key=lambda r: r["path"])
        _atomic_write(manifest_path, canonical_json(manifest) + b"\n")
        self._index += 1
        self._partial = self.root / f"{self.prefix}-{self._index:05d}.jsonl.partial"
        self._count = 0
        self._stream = self._partial.open("ab")

    def close(self) -> None:
        self._finalize()
        if not self._stream.closed:
            self._stream.close()
        if self._partial.exists() and self._partial.stat().st_size == 0:
            self._partial.unlink()

    def __enter__(self) -> "ResumableShardWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def iter_shards(root: Path, *, prefix: str = "records", verify: bool = True) -> Iterator[dict[str, Any]]:
    root = Path(root)
    manifest = json.loads((root / f"{prefix}-manifest.json").read_text(encoding="utf-8"))
    for shard in manifest["shards"]:
        path = root / shard["path"]
        if verify and _sha256_file(path) != shard["sha256"]:
            raise IOError(f"shard checksum mismatch: {path}")
        count = 0
        with path.open("rb") as stream:
            for line in stream:
                count += 1
                yield validate_record(json.loads(line))
        if count != shard["records"]:
            raise IOError(f"shard record count mismatch: {path}")


def read_shards(root: Path, *, prefix: str = "records", verify: bool = True) -> list[dict[str, Any]]:
    """Convenience loader for tests and small manifests; production can stream."""
    return list(iter_shards(root, prefix=prefix, verify=verify))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

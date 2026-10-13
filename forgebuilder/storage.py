"""Immutable content-addressed storage with restricted-source references."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .schema import canonical_json


class ContentAddressedStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def put(self, payload: bytes) -> dict[str, Any]:
        digest = hashlib.sha256(payload).hexdigest()
        target = self.root / digest[:2] / digest[2:]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _sha256_file(target) != digest:
                raise IOError(f"immutable object is corrupt: {target}")
            return {"sha256": digest, "size_bytes": target.stat().st_size, "path": target.relative_to(self.root).as_posix()}
        fd, temp_name = tempfile.mkstemp(prefix=".object-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if _sha256_file(Path(temp_name)) != digest:
                raise IOError("temporary object failed its digest check")
            try:
                os.link(temp_name, target)
            except FileExistsError:
                if _sha256_file(target) != digest:
                    raise IOError(f"immutable object raced with corrupt data: {target}")
            return {"sha256": digest, "size_bytes": len(payload), "path": target.relative_to(self.root).as_posix()}
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def get(self, digest: str) -> bytes:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("digest must be a lowercase SHA-256 value")
        path = self.root / digest[:2] / digest[2:]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise IOError(f"content-addressed object failed verification: {path}")
        return payload

    def put_json(self, value: Any) -> dict[str, Any]:
        return self.put(canonical_json(value))

    @staticmethod
    def source_reference(*, source_url: str | None, commit_sha: str | None, license_id: str | None,
                         expected_sha256: str, extraction_recipe: Mapping[str, Any] | None = None,
                         redistributable: bool, swhid: str | None = None) -> dict[str, Any]:
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 value")
        return {"source_url": source_url, "commit_sha": commit_sha, "license": license_id,
                "expected_sha256": expected_sha256, "swhid": swhid,
                "extraction_recipe": dict(extraction_recipe or {}),
                "redistributable": bool(redistributable), "payload_retained": False}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

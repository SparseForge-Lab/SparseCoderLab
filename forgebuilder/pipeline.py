"""Model-neutral generator, critic, verifier, batching, and queue interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Mapping, Protocol, Sequence

from .schema import canonical_json

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@contextmanager
def _transaction(db: sqlite3.Connection):
    try:
        with db:
            yield db
    finally:
        db.close()


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    generator_model: str
    task_family: str
    context_tokens: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class GenerationCandidate:
    request_id: str
    generator_model: str
    generator_version: str
    text: str
    latency_seconds: float | None = None
    token_count: int | None = None
    tool_trace_hash: str | None = None


class Generator(Protocol):
    model_id: str
    version: str

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationCandidate]: ...


@dataclass(frozen=True)
class CriticResult:
    verdict: str
    confidence: float
    functional_issue: bool = False
    hallucination: bool = False
    scope_violation: bool = False
    stop_issue: bool = False
    reasoning_issue: bool = False
    test_gaming: bool = False
    needs_deeper_review: bool = False
    issue_summary: str = ""
    suggested_fix: str = ""
    critic_model: str = ""
    critic_version: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in {"PASS", "REVISE", "FAIL"}:
            raise ValueError("critic verdict must be PASS, REVISE, or FAIL")
        if not 0 <= self.confidence <= 1:
            raise ValueError("critic confidence must be between zero and one")


class Critic(Protocol):
    model_id: str
    version: str

    def critique(self, *, task: Mapping[str, Any], candidate: str,
                  evidence: Mapping[str, Any], profile: str = "normal") -> CriticResult: ...


class Verifier(Protocol):
    verifier_id: str
    version: str

    def verify(self, *, task: Mapping[str, Any], candidate: str,
               environment: Mapping[str, Any]) -> "VerificationResult": ...


@dataclass(frozen=True)
class VerificationResult:
    status: str
    verifier_id: str
    verifier_version: str
    test_hash: str | None = None
    environment_hash: str | None = None
    passed: int = 0
    failed: int = 0
    timed_out: bool = False
    evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "FAIL", "ERROR", "NOT_RUN"}:
            raise ValueError("invalid verification status")
        if min(self.passed, self.failed) < 0:
            raise ValueError("test counts cannot be negative")
        if self.status == "PASS" and self.failed:
            raise ValueError("PASS cannot include failing tests")
        for name, digest in (("test_hash", self.test_hash), ("environment_hash", self.environment_hash)):
            if digest is not None and not _SHA256.fullmatch(digest):
                raise ValueError(f"{name} must be null or a lowercase SHA-256 digest")
        if self.status == "PASS" and self.timed_out:
            raise ValueError("timed out verification cannot pass")


def accept_posttraining_candidate(*, verification: VerificationResult, critic: CriticResult,
                                  candidate_generated_by_critic: bool = False,
                                  critic_context_reset: bool = False) -> bool:
    """Require objective success and an independent semantic review."""
    independent = not candidate_generated_by_critic or critic_context_reset
    objective = (verification.status == "PASS" and verification.passed > 0
                 and bool(verification.test_hash) and bool(verification.environment_hash))
    return bool(objective and independent and critic.verdict == "PASS")


def schedule_batches(requests: Sequence[GenerationRequest], *, batch_size: int) -> list[list[GenerationRequest]]:
    """Group deterministically by model and task/context profile to avoid reloads."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    groups: dict[tuple[str, str, int], list[GenerationRequest]] = {}
    for row in requests:
        context_bucket = 1 << max(0, (max(1, row.context_tokens) - 1).bit_length())
        groups.setdefault((row.generator_model, row.task_family, context_bucket), []).append(row)
    batches = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: row.request_id)
        batches.extend([rows[i:i + batch_size] for i in range(0, len(rows), batch_size)])
    return batches


class CriticCache:
    """SQLite cache keyed by all inputs that can change a critic judgment."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _transaction(self._connect()) as db:
            db.execute("CREATE TABLE IF NOT EXISTS critic_cache (cache_key TEXT PRIMARY KEY, result_json BLOB NOT NULL, created REAL NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @staticmethod
    def key(*, task: Mapping[str, Any], context: Mapping[str, Any], candidate: str,
            patch: str, evidence: Mapping[str, Any], critic_model: str,
            critic_version: str, critic_config: Mapping[str, Any], prompt_version: str) -> str:
        identity = {"task": task, "context": context, "candidate": candidate, "patch": patch,
                    "evidence": evidence, "critic_model": critic_model,
                    "critic_version": critic_version, "critic_config": critic_config,
                    "prompt_version": prompt_version}
        return hashlib.sha256(canonical_json(identity)).hexdigest()

    def get(self, cache_key: str) -> CriticResult | None:
        with _transaction(self._connect()) as db:
            row = db.execute("SELECT result_json FROM critic_cache WHERE cache_key=?", (cache_key,)).fetchone()
        if row is None:
            return None
        return CriticResult(**json.loads(row[0]))

    def put(self, cache_key: str, result: CriticResult) -> None:
        if not result.critic_model or not result.critic_version:
            raise ValueError("cached critic results require model and version identities")
        payload = canonical_json(asdict(result))
        with _transaction(self._connect()) as db:
            db.execute("INSERT OR REPLACE INTO critic_cache(cache_key,result_json,created) VALUES(?,?,?)",
                       (cache_key, payload, time.time()))


class TaskQueue:
    """Small transactional queue whose expired leases return to pending state."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _transaction(self._connect()) as db:
            db.execute("CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, payload BLOB NOT NULL, status TEXT NOT NULL, lease_owner TEXT, lease_until REAL, updated REAL NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level="IMMEDIATE")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def enqueue(self, task_id: str, payload: Mapping[str, Any]) -> bool:
        if not task_id:
            raise ValueError("task_id is required")
        now = time.time()
        with _transaction(self._connect()) as db:
            cur = db.execute("INSERT OR IGNORE INTO tasks(task_id,payload,status,updated) VALUES(?,?,?,?)",
                             (task_id, canonical_json(payload), "pending", now))
            return cur.rowcount == 1

    def lease(self, owner: str, *, limit: int = 1, lease_seconds: float = 300) -> list[dict[str, Any]]:
        if not owner or limit < 1 or lease_seconds <= 0:
            raise ValueError("owner, positive limit, and positive lease duration are required")
        now = time.time()
        rows = []
        with _transaction(self._connect()) as db:
            db.execute("UPDATE tasks SET status='pending', lease_owner=NULL, lease_until=NULL, updated=? WHERE status='leased' AND lease_until<?", (now, now))
            selected = db.execute("SELECT task_id,payload FROM tasks WHERE status='pending' ORDER BY task_id LIMIT ?", (limit,)).fetchall()
            for task_id, payload in selected:
                db.execute("UPDATE tasks SET status='leased',lease_owner=?,lease_until=?,updated=? WHERE task_id=? AND status='pending'",
                           (owner, now + lease_seconds, now, task_id))
                rows.append({"task_id": task_id, "payload": json.loads(payload)})
        return rows

    def complete(self, task_id: str, owner: str) -> None:
        with _transaction(self._connect()) as db:
            cur = db.execute("UPDATE tasks SET status='done',lease_owner=NULL,lease_until=NULL,updated=? WHERE task_id=? AND status='leased' AND lease_owner=?",
                             (time.time(), task_id, owner))
            if cur.rowcount != 1:
                raise ValueError("task is not leased by this owner")

    def recover(self) -> dict[str, int]:
        now = time.time()
        with _transaction(self._connect()) as db:
            db.execute("UPDATE tasks SET status='pending',lease_owner=NULL,lease_until=NULL,updated=? WHERE status='leased' AND lease_until<?", (now, now))
            rows = db.execute("SELECT status,count(*) FROM tasks GROUP BY status").fetchall()
        return {status: count for status, count in rows}


class GeneratorQualityTracker:
    """Append-only aggregate counts by generator and task family."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _transaction(sqlite3.connect(self.path)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS stats (model TEXT, family TEXT, metric TEXT, value REAL NOT NULL, n INTEGER NOT NULL, PRIMARY KEY(model,family,metric))")

    def add(self, model: str, family: str, metric: str, value: float = 1.0) -> None:
        with _transaction(sqlite3.connect(self.path)) as db:
            db.execute("INSERT INTO stats VALUES(?,?,?,?,1) ON CONFLICT(model,family,metric) DO UPDATE SET value=value+excluded.value,n=n+1",
                       (model, family, metric, float(value)))

    def report(self) -> list[dict[str, Any]]:
        with _transaction(sqlite3.connect(self.path)) as db:
            rows = db.execute("SELECT model,family,metric,value,n FROM stats ORDER BY model,family,metric").fetchall()
        return [{"model": m, "task_family": f, "metric": metric, "total": value, "count": n,
                 "mean": value / n if n else None} for m, f, metric, value, n in rows]

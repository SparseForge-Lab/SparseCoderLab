"""ForgeBuilder task contracts for objective repository repair examples."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Protocol, Sequence

from .pipeline import CriticResult, VerificationResult
from .schema import canonical_json, sha256_bytes, validate_record


class RepositoryReconstructor(Protocol):
    def checkout(self, repository: str, commit_sha: str, environment: Mapping[str, Any]) -> Any: ...


class IsolatedRunner(Protocol):
    def run(self, checkout: Any, command: Sequence[str], *, timeout_seconds: int) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ForgeTestEvidence:
    test_id: str
    before_status: str
    after_status: str
    test_hash: str


@dataclass(frozen=True)
class ForgeTask:
    repository: str
    base_commit: str
    fixed_commit: str
    issue_id: str
    task_description: str
    environment_hash: str
    fail_to_pass: tuple[ForgeTestEvidence, ...]
    pass_to_pass: tuple[ForgeTestEvidence, ...]
    patch_sha256: str
    source_provenance: Mapping[str, Any]


def validate_forge_task(task: ForgeTask) -> dict[str, Any]:
    if not task.repository or not task.issue_id:
        raise ValueError("repository, exact commits, and issue_id are required")
    for commit in (task.base_commit, task.fixed_commit):
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
            raise ValueError("base_commit and fixed_commit must be full lowercase Git object IDs")
    if not task.task_description.strip():
        raise ValueError("task description cannot be empty")
    for name, digest in (("environment_hash", task.environment_hash), ("patch_sha256", task.patch_sha256)):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    for row in (*task.fail_to_pass, *task.pass_to_pass):
        if len(row.test_hash) != 64 or any(c not in "0123456789abcdef" for c in row.test_hash):
            raise ValueError("test_hash must be a lowercase SHA-256 digest")
    if not task.fail_to_pass:
        raise ValueError("at least one fail-to-pass test is required")
    if not task.pass_to_pass:
        raise ValueError("at least one pass-to-pass regression test is required")
    if any(row.before_status != "FAIL" or row.after_status != "PASS" for row in task.fail_to_pass):
        raise ValueError("fail-to-pass tests must fail at base and pass at fixed commit")
    if any(row.before_status != "PASS" or row.after_status != "PASS" for row in task.pass_to_pass):
        raise ValueError("regression tests must pass at both commits")
    if not task.source_provenance:
        raise ValueError("task provenance is required")
    return {"schema_version": 1, "repository": task.repository, "base_commit": task.base_commit,
            "fixed_commit": task.fixed_commit, "issue_id": task.issue_id,
            "task_description": task.task_description, "environment_hash": task.environment_hash,
            "fail_to_pass": [row.__dict__ for row in task.fail_to_pass],
            "pass_to_pass": [row.__dict__ for row in task.pass_to_pass],
            "patch_sha256": task.patch_sha256, "source_provenance": dict(task.source_provenance)}


def check_patch_leakage(task_description: str, changed_lines: Sequence[str], *, min_chars: int = 24) -> dict[str, Any]:
    """Flag exact patch-line leakage; intentionally does not claim semantic completeness."""
    description = task_description.casefold()
    hits = sorted({line.strip() for line in changed_lines
                   if len(line.strip()) >= min_chars and line.strip().casefold() in description})
    return {"status": "flagged" if hits else "clear_exact_line_screen", "matched_lines": hits,
            "method": "normalized_exact_changed_line", "semantic_leakage_claim": False}


def accept_candidate(*, fail_to_pass: VerificationResult, pass_to_pass: VerificationResult,
                     critic: CriticResult, diff_inspected: bool,
                     candidate_generated_by_critic: bool = False,
                     critic_context_reset: bool = False) -> bool:
    """Teacher opinion alone cannot accept a candidate."""
    objective_pass = all(row.status == "PASS" and row.passed > 0 and row.test_hash and row.environment_hash
                         for row in (fail_to_pass, pass_to_pass))
    same_environment = fail_to_pass.environment_hash == pass_to_pass.environment_hash
    independent_critic = bool(critic.critic_model and critic.critic_version
                              and (not candidate_generated_by_critic or critic_context_reset))
    return bool(objective_pass and same_environment and independent_critic
                and critic.verdict == "PASS" and diff_inspected)


def normalize_candidate_record(*, task: ForgeTask, candidate_patch: str,
                               fail_to_pass: VerificationResult, pass_to_pass: VerificationResult,
                               critic: CriticResult, diff_inspected: bool,
                               generator_model: str | None = None,
                               generator_version: str | None = None,
                               tool_trace_hash: str | None = None,
                               critic_context_reset: bool = False) -> dict[str, Any]:
    """Normalize a repository candidate; failed candidates remain explicit rejects."""
    task_record = validate_forge_task(task)
    accepted = accept_candidate(fail_to_pass=fail_to_pass, pass_to_pass=pass_to_pass,
                                critic=critic, diff_inspected=diff_inspected,
                                candidate_generated_by_critic=bool(generator_model and generator_model == critic.critic_model),
                                critic_context_reset=critic_context_reset)
    objective_pass = all(row.status == "PASS" and row.passed > 0 and row.test_hash and row.environment_hash
                         for row in (fail_to_pass, pass_to_pass))
    source = {**dict(task.source_provenance), "repository": task.repository,
              "base_commit": task.base_commit, "fixed_commit": task.fixed_commit,
              "issue_id": task.issue_id, "environment_hash": task.environment_hash}
    patch_hash = hashlib.sha256(candidate_patch.encode("utf-8")).hexdigest()
    content = {"task_description": task.task_description, "candidate_patch": candidate_patch,
               "task_contract": task_record, "fail_to_pass": fail_to_pass.__dict__,
               "pass_to_pass": pass_to_pass.__dict__, "critic": critic.__dict__,
               "diff_inspected": bool(diff_inspected)}
    content_hash = hashlib.sha256(canonical_json(content)).hexdigest()
    rejection = None
    if not accepted:
        if not objective_pass:
            rejection = "objective_tests_failed_or_missing"
        elif not diff_inspected:
            rejection = "candidate_diff_not_inspected"
        elif generator_model and generator_model == critic.critic_model and not critic_context_reset:
            rejection = "critic_context_not_independent"
        elif not critic.critic_model or not critic.critic_version:
            rejection = "critic_identity_missing"
        else:
            rejection = f"critic_{critic.verdict.casefold()}"
    combined_test_hash = (sha256_bytes(canonical_json({"fail_to_pass": fail_to_pass.test_hash,
                                                       "pass_to_pass": pass_to_pass.test_hash}))
                          if fail_to_pass.test_hash and pass_to_pass.test_hash else None)
    row = {
        "schema_version": 1, "source_id": f"forge:{task.repository}:{task.issue_id}:{task.base_commit}",
        "source_type": "forge_repository_candidate", "source_url": source.get("source_url"),
        "repo": task.repository, "commit_sha": task.fixed_commit,
        "timestamp": source.get("timestamp"), "license": source.get("license"),
        "language": source.get("language"), "natural_language": source.get("natural_language"),
        "file_path": source.get("file_path"), "content_sha256": content_hash,
        "repo_id": task.repository, "near_dedup_cluster": source.get("near_dedup_cluster"),
        "token_count": None, "quality_tier": "verified_posttraining" if accepted else "rejected_candidate",
        "task_family": "bug_fix", "difficulty": source.get("difficulty"),
        "real_or_generated": "derived", "data_stage": "posttraining",
        "generator_model": generator_model, "generator_version": generator_version,
        "generator_prompt_version": source.get("generator_prompt_version"),
        "critic_model": critic.critic_model or None, "critic_version": critic.critic_version or None,
        "execution_verified": bool(objective_pass), "compiler_verified": None, "test_verified": bool(objective_pass),
        "test_hash": combined_test_hash,
        "environment_hash": task.environment_hash,
        "tool_trace_hash": tool_trace_hash, "reasoning_type": "evidence_action_verify",
        "stop_should_happen": True, "hallucination_category": source.get("hallucination_category"),
        "split": "unassigned", "eval_quarantine_status": "not_screened",
        "source_provenance": source, "transformation_recipe": "issue_to_verified_patch",
        "parent_sample_ids": list(source.get("parent_sample_ids", [])),
        "student_model": None, "student_version": None, "student_failure_type": None,
        "recovery_source": None, "recovery_verified": None,
        "accepted": accepted, "rejection_reason": rejection,
        "created_at": source.get("created_at"), "payload": content,
        "candidate_patch_sha256": patch_hash,
        "critic_verdict": critic.verdict,
        "diff_inspected": bool(diff_inspected),
    }
    return validate_record(row)


def forge_sample_id(task: ForgeTask, candidate_patch_sha256: str) -> str:
    if len(candidate_patch_sha256) != 64 or any(c not in "0123456789abcdef" for c in candidate_patch_sha256):
        raise ValueError("candidate_patch_sha256 must be a lowercase SHA-256 digest")
    identity = {"repo": task.repository, "base": task.base_commit, "fixed": task.fixed_commit,
                "issue": task.issue_id, "candidate_patch": candidate_patch_sha256,
                "environment": task.environment_hash}
    return "forge-" + sha256_bytes(canonical_json(identity))

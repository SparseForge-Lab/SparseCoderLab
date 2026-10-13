import hashlib
import json
import sqlite3

import pytest

from forgebuilder.forge import (ForgeTask, ForgeTestEvidence, accept_candidate,
                                     check_patch_leakage, normalize_candidate_record,
                                     validate_forge_task)
from forgebuilder.manifest import (EvalQuarantine, assign_repository_split,
                                       read_manifest, write_manifest)
from forgebuilder.pipeline import (CriticCache, CriticResult, GenerationRequest,
                                       GeneratorQualityTracker, TaskQueue, VerificationResult,
                                       accept_posttraining_candidate,
                                       schedule_batches)
from forgebuilder.recovery import failure_priority, validate_recovery, validate_student_failure
from forgebuilder.schema import SchemaError, stable_sample_id, validate_record
from forgebuilder.shards import ResumableShardWriter, read_shards
from forgebuilder.stop_scope import stop_scope_metrics, validate_stop_continue_pair
from forgebuilder.storage import ContentAddressedStore
from forgebuilder.trajectory import normalize_trajectory


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sample(text="tiny synthetic fixture", **updates):
    row = {
        "schema_version": 1,
        "source_id": "fixture:one",
        "source_type": "synthetic_fixture",
        "source_url": None,
        "repo": None,
        "commit_sha": None,
        "timestamp": None,
        "license": None,
        "language": "Python",
        "natural_language": ["en"],
        "file_path": None,
        "content_sha256": digest(text),
        "repo_id": "fixture/repo",
        "near_dedup_cluster": None,
        "token_count": None,
        "quality_tier": "unreviewed",
        "task_family": "file_implementation",
        "difficulty": None,
        "real_or_generated": "generated",
        "data_stage": "posttraining",
        "generator_model": "fixture-model",
        "generator_version": "fixture-v1",
        "generator_prompt_version": "fixture-prompt-v1",
        "critic_model": None,
        "critic_version": None,
        "execution_verified": None,
        "compiler_verified": None,
        "test_verified": None,
        "test_hash": None,
        "environment_hash": None,
        "tool_trace_hash": None,
        "reasoning_type": None,
        "stop_should_happen": None,
        "hallucination_category": None,
        "split": "unassigned",
        "eval_quarantine_status": "not_screened",
        "source_provenance": {"fixture": "tiny-local-test"},
        "transformation_recipe": None,
        "parent_sample_ids": [],
        "student_model": None,
        "student_version": None,
        "student_failure_type": None,
        "recovery_source": None,
        "recovery_verified": None,
        "accepted": None,
        "rejection_reason": None,
        "created_at": "2026-01-01T00:00:00Z",
        "payload_text": text,
        **updates,
    }
    return row


def test_schema_stable_id_provenance_and_license_are_explicit():
    row = validate_record(sample())
    multimodal = validate_record(sample(modality="image_text", input_modalities=["image", "text"],
                                        output_modalities=["code"], asset_references=[{"sha256": "a"*64}]))
    assert multimodal["input_modalities"] == ["image", "text"]
    same = validate_record({**sample(), "created_at": "2026-02-01T00:00:00Z", "quality_tier": "reviewed"})
    assert row["sample_id"] == stable_sample_id(same)
    assert row["license"] is None
    with pytest.raises(SchemaError, match="license"):
        validate_record({k: v for k, v in sample().items() if k != "license"})
    with pytest.raises(SchemaError, match="provenance"):
        validate_record({k: v for k, v in sample().items() if k != "source_provenance"})
    with pytest.raises(SchemaError, match="rejection_reason"):
        validate_record(sample(accepted=False))


def test_manifest_roundtrip_order_hash_and_repo_split(tmp_path):
    rows = [sample("second"), sample("first", source_id="fixture:two")]
    report = write_manifest(tmp_path / "manifest.jsonl", rows)
    second_report = write_manifest(tmp_path / "manifest-reordered.jsonl", reversed(rows))
    actual = read_manifest(tmp_path / "manifest.jsonl", verify_hash=report["sha256"])
    assert report["sha256"] == second_report["sha256"]
    assert [r["sample_id"] for r in actual] == sorted(r["sample_id"] for r in actual)
    assert assign_repository_split("owner/repo", seed=42) == assign_repository_split("owner/repo", seed=42)


def test_eval_quarantine_is_explicitly_a_lexical_screen():
    prompt = "implement this synthetic checksum validation routine with stable errors"
    screen = EvalQuarantine([{"benchmark_id": "fixture-eval", "prompt": prompt}], shingle_width=3)
    hit = screen.inspect(prompt)
    clear = screen.inspect("write a parser that accepts one short token")
    assert hit["status"] == "flagged" and hit["matches"][0]["benchmark_id"] == "fixture-eval"
    assert clear["status"] == "clear_lexical_screen"
    assert clear["universal_contamination_claim"] is False
    assert screen.annotate(sample(), prompt)["eval_quarantine_status"] == "flagged"


def test_content_addressed_storage_reuses_immutable_object(tmp_path):
    store = ContentAddressedStore(tmp_path / "objects")
    first = store.put(b"tiny immutable fixture")
    second = store.put(b"tiny immutable fixture")
    assert first == second
    assert store.get(first["sha256"]) == b"tiny immutable fixture"
    ref = store.source_reference(source_url="https://example.invalid/repo", commit_sha="abc",
                                license_id=None, expected_sha256=first["sha256"], redistributable=False)
    assert ref["payload_retained"] is False and ref["license"] is None


def test_batching_groups_model_and_task_without_reordering_reproducibility():
    requests = [
        GenerationRequest("b", "ornith", "repo_fix", 50_000, {}),
        GenerationRequest("a", "ornith", "repo_fix", 60_000, {}),
        GenerationRequest("c", "bonsai", "file_implementation", 8_000, {}),
    ]
    batches = schedule_batches(requests, batch_size=4)
    assert [[r.request_id for r in b] for b in batches] == [["c"], ["a", "b"]]


def test_critic_cache_identity_covers_context_patch_evidence_and_version(tmp_path):
    cache = CriticCache(tmp_path / "critic.sqlite")
    key = cache.key(task={"id": "t1"}, context={"repo": "r"}, candidate="candidate", patch="patch",
                    evidence={"tests": "pass"}, critic_model="flash", critic_version="full-v1",
                    critic_config={"ctx": 65536},
                    prompt_version="v1")
    changed = cache.key(task={"id": "t1"}, context={"repo": "r"}, candidate="candidate", patch="patch",
                        evidence={"tests": "fail"}, critic_model="flash", critic_version="full-v1",
                        critic_config={"ctx": 65536},
                        prompt_version="v1")
    changed_version = cache.key(task={"id": "t1"}, context={"repo": "r"}, candidate="candidate", patch="patch",
                                evidence={"tests": "pass"}, critic_model="flash", critic_version="full-v2",
                                critic_config={"ctx": 65536}, prompt_version="v1")
    result = CriticResult("PASS", 0.91, critic_model="flash", critic_version="v1")
    assert key != changed and key != changed_version and cache.get(key) is None
    cache.put(key, result)
    assert cache.get(key) == result


def test_queue_recovers_expired_lease_and_completes_owner_checked_task(tmp_path):
    queue = TaskQueue(tmp_path / "queue.sqlite")
    assert queue.enqueue("task-1", {"family": "bug_fix"})
    assert not queue.enqueue("task-1", {"family": "different"})
    assert queue.lease("worker-a")[0]["payload"]["family"] == "bug_fix"
    with sqlite3.connect(queue.path) as db:
        db.execute("UPDATE tasks SET lease_until=0 WHERE task_id='task-1'")
    assert queue.recover() == {"pending": 1}
    assert queue.lease("worker-b")[0]["task_id"] == "task-1"
    queue.complete("task-1", "worker-b")
    assert queue.recover() == {"done": 1}


def test_generator_quality_is_aggregated_per_model_and_task_family(tmp_path):
    tracker = GeneratorQualityTracker(tmp_path / "generator_stats.sqlite")
    tracker.add("bonsai-fixture", "bug_fix", "objective_pass_rate", 1.0)
    tracker.add("bonsai-fixture", "bug_fix", "objective_pass_rate", 0.0)
    row = tracker.report()[0]
    assert row["count"] == 2 and row["mean"] == 0.5


def test_resumable_shard_writer_recovers_partial_and_roundtrips(tmp_path):
    root = tmp_path / "shards"
    writer = ResumableShardWriter(root, max_records=2)
    writer.write(sample("one"))
    writer._stream.close()  # simulate process loss before finalizing the partial file
    resumed = ResumableShardWriter(root, max_records=2)
    resumed.write(sample("two", source_id="fixture:two"))
    resumed.write(sample("three", source_id="fixture:three"))
    resumed.close()
    rows = read_shards(root)
    assert len(rows) == 3
    manifest = json.loads((root / "records-manifest.json").read_text())
    assert [row["records"] for row in manifest["shards"]] == [2, 1]


def test_shard_writer_recovers_finalized_file_missing_from_manifest(tmp_path):
    root = tmp_path / "shards"
    writer = ResumableShardWriter(root, max_records=4)
    writer.write(sample("one"))
    writer._stream.close()
    writer._partial.rename(writer._partial.with_suffix(""))
    resumed = ResumableShardWriter(root, max_records=4)
    resumed.close()
    assert len(read_shards(root)) == 1


def test_forge_gates_require_pre_post_tests_critic_and_diff_review():
    h = "a" * 64
    task = ForgeTask("owner/repo", "1" * 40, "2" * 40, "issue-1", "Fix the checksum parser.",
                     h, (ForgeTestEvidence("fix", "FAIL", "PASS", h),),
                     (ForgeTestEvidence("regression", "PASS", "PASS", h),), h,
                     {"source_url": "https://example.invalid", "license": "MIT"})
    assert validate_forge_task(task)["repository"] == "owner/repo"
    f2p = VerificationResult("PASS", "pytest", "9", h, h, passed=1)
    p2p = VerificationResult("PASS", "pytest", "9", h, h, passed=2)
    critic = CriticResult("PASS", 0.9, critic_model="flash", critic_version="v1")
    assert accept_candidate(fail_to_pass=f2p, pass_to_pass=p2p, critic=critic, diff_inspected=True)
    assert not accept_candidate(fail_to_pass=f2p, pass_to_pass=p2p,
                                critic=CriticResult("PASS", 1.0), diff_inspected=False)
    assert not accept_candidate(fail_to_pass=VerificationResult("PASS", "pytest", "9"), pass_to_pass=p2p,
                                critic=critic, diff_inspected=True)
    normalized = normalize_candidate_record(task=task, candidate_patch="patch text", fail_to_pass=f2p,
        pass_to_pass=p2p, critic=CriticResult("PASS", 0.95, critic_model="flash", critic_version="v1"),
        diff_inspected=True, generator_model="fixture", generator_version="v1")
    assert normalized["accepted"] and normalized["test_verified"]
    rejected = normalize_candidate_record(task=task, candidate_patch="bad patch", fail_to_pass=f2p,
        pass_to_pass=p2p, critic=CriticResult("FAIL", 0.95, critic_model="flash", critic_version="v1"),
        diff_inspected=True)
    assert not rejected["accepted"] and rejected["rejection_reason"] == "critic_fail"
    same_model = normalize_candidate_record(task=task, candidate_patch="patch text", fail_to_pass=f2p,
        pass_to_pass=p2p, critic=CriticResult("PASS", 0.95, critic_model="flash", critic_version="v1"),
        diff_inspected=True, generator_model="flash", generator_version="v1")
    assert not same_model["accepted"] and same_model["rejection_reason"] == "critic_context_not_independent"
    assert check_patch_leakage("Fix the checksum parser.", ["if checksum != expected_value: return False"])["status"].startswith("clear")
    with pytest.raises(ValueError, match="fail at base"):
        validate_forge_task(ForgeTask("owner/repo", "1"*40, "2"*40, "i", "Fix it", h,
            (ForgeTestEvidence("fix", "PASS", "PASS", h),),
            (ForgeTestEvidence("regression", "PASS", "PASS", h),), h, {"url": "known"}))


def test_student_failure_masks_known_bad_actions_and_links_verified_recovery():
    failure = validate_student_failure({"sample_id": "failure-1", "student_model": "student",
        "student_version": "v1", "failure_type": "invented_api", "state": {"repo": "current"},
        "evidence": {"symbol_search": "no result"}, "wrong_hypothesis": "the API exists",
        "environment": {"python": "3.12"}, "known_bad_actions": [{"action": "call_missing_api"}],
        "recurrence_count": 2, "confidence_before_failure": 0.95, "objectively_verified": True})
    assert failure["known_bad_actions"][0]["loss_mask"] == 0.0
    assert failure["known_bad_actions"][0]["target_weight"] == 0.0
    assert failure_priority(failure) > 1.0
    result = VerificationResult("PASS", "pytest", "9", "a"*64, "b"*64, passed=3)
    recovery = validate_recovery(parent_sample_id="failure-1", failure=failure, diagnosis="symbol is absent",
        recovery_actions=[{"action": "inspect exports"}], verification=result,
        final_result="tests pass", recovery_source="independent_recovery")
    assert recovery["accepted"] and recovery["recovery_verified"]
    assert recovery["recurrence_count"] == 2 and recovery["confidence_before_failure"] == 0.95
    with pytest.raises(ValueError, match="parent_sample_id"):
        validate_recovery(parent_sample_id="wrong", failure=failure, diagnosis="diagnosis",
            recovery_actions=[{"action": "fix"}], verification=result, final_result="done",
            recovery_source="teacher")


def test_structured_trajectory_masks_known_bad_actions_and_hashes_trace():
    trace = normalize_trajectory([
        {"type": "EVIDENCE", "content": "export search did not find symbol"},
        {"type": "WRONG_HYPOTHESIS", "content": "symbol may exist in another module"},
        {"type": "ACTION", "content": "call unverified symbol", "known_bad": True},
        {"type": "DIAGNOSIS", "content": "repository exports do not contain the symbol"},
        {"type": "VERIFY", "content": "three tests pass", "evidence_refs": ["test-sha"]},
        {"type": "FINAL", "content": "use the documented function"},
    ])
    assert trace["events"][2]["loss_mask"] == 0.0
    assert trace["events"][2]["target_weight"] == 0.0
    assert trace["events"][1]["loss_mask"] == 0.0
    assert len(trace["tool_trace_hash"]) == 64


def test_stop_continue_pair_and_scope_measurements():
    stop = {"pair_id": "pair-1", "sample_id": "stop-1", "task_family": "stop_scope",
            "requested_scope": "one function", "stop_should_happen": True}
    continuation = {"pair_id": "pair-1", "sample_id": "continue-1", "task_family": "stop_scope",
                    "requested_scope": "two functions", "stop_should_happen": False}
    validate_stop_continue_pair(stop, continuation)
    metrics = stop_scope_metrics(target_complete=True, natural_stop=False, cap_reached=True,
                                 tokens_after_target=42, repetitions_after_target=8,
                                 continuation_required=False)
    assert metrics["tokens_after_target"] == 42
    assert metrics["repetition_after_target"] == 8
    assert metrics["cap_reached"] is True and metrics["premature_stop"] is False
    with pytest.raises(ValueError, match="stop_should_happen"):
        validate_stop_continue_pair(stop, {**continuation, "stop_should_happen": True})


def test_verifier_result_rejects_inconsistent_pass():
    result = VerificationResult("PASS", "fixture", "v1", "a"*64, "b"*64, passed=2)
    assert result.status == "PASS"
    assert accept_posttraining_candidate(verification=result, critic=CriticResult("PASS", 0.9))
    assert not accept_posttraining_candidate(verification=result, critic=CriticResult("PASS", 0.9),
                                             candidate_generated_by_critic=True)
    assert accept_posttraining_candidate(verification=result, critic=CriticResult("PASS", 0.9),
                                        candidate_generated_by_critic=True, critic_context_reset=True)
    with pytest.raises(ValueError, match="PASS cannot"):
        VerificationResult("PASS", "fixture", "v1", passed=1, failed=1)

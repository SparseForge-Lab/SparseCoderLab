"""Coverage and rights accounting over durable pilot review snapshots."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .pipeline import CriticResult, VerificationResult, accept_posttraining_candidate
from .schema import sha256_bytes


def rate(numerator,denominator):
    return numerator/denominator if denominator else None


def inspect_pilot(pilot):
    pilot=Path(pilot)
    preparation=json.loads((pilot/'preparation.json').read_bytes())
    payload=(pilot/'candidates.jsonl').read_bytes()
    if sha256_bytes(payload)!=preparation['candidate_sha256']:
        raise ValueError('pilot candidate identity changed')
    candidates={}
    for line in payload.splitlines():
        row=json.loads(line)
        if row['sample_id'] in candidates:
            raise ValueError('duplicate candidate identity')
        candidates[row['sample_id']]=row
    path=pilot/'reviews.jsonl'
    raw=path.read_bytes() if path.exists() else b''
    # The live writer appends and fsyncs one JSON line at a time. An unfinished
    # final line remains pending; never convert it into a semantic failure.
    complete=raw if raw.endswith(b'\n') else raw[:raw.rfind(b'\n')+1]
    reviews={}
    for line in complete.splitlines():
        review=json.loads(line)
        identity=review['sample_id']
        if identity not in candidates or identity in reviews:
            raise ValueError('unknown or duplicate reviewed identity')
        row=candidates[identity]
        if review['task_family']!=row['task_family']:
            raise ValueError('review task family changed')
        objective=VerificationResult(**review['objective_verification'])
        critic=CriticResult(**review['critic'])
        accepted=accept_posttraining_candidate(verification=objective,critic=critic)
        if bool(review['accepted'])!=accepted:
            raise ValueError('saved acceptance contradicts objective/critic evidence')
        if accepted and (not review.get('fresh_context') or any(getattr(critic,k) for k in
                ('functional_issue','hallucination','scope_violation','stop_issue','reasoning_issue','test_gaming','needs_deeper_review'))):
            raise ValueError('accepted review has unresolved issues or no fresh context')
        reviews[identity]=review
    accepted=[row for key,row in candidates.items() if reviews.get(key,{}).get('accepted')]
    pair_counts=Counter(row['pair_id'] for row in accepted if 'pair_id' in row)
    complete_pairs=[row for row in accepted if 'pair_id' not in row or pair_counts[row['pair_id']]==2]
    families=defaultdict(list)
    for row in candidates.values(): families[row['task_family']].append(row)
    metrics={}
    for family,rows in sorted(families.items()):
        completed=[reviews[r['sample_id']] for r in rows if r['sample_id'] in reviews]
        count=len(completed)
        objective_pass=sum(r['objective_verification']['status']=='PASS' for r in completed)
        critic_fail=sum(r['critic']['verdict']=='FAIL' for r in completed)
        accepted_count=sum(r['accepted'] for r in completed)
        flags={flag:sum(bool(r['critic'][flag]) for r in completed)
               for flag in ('hallucination','scope_violation','stop_issue')}
        metrics[family]=dict(candidates=len(rows),reviewed=count,pending=len(rows)-count,
            objective_passes=objective_pass,objective_pass_rate=rate(objective_pass,count),
            critic_failures=critic_fail,critic_failure_rate=rate(critic_fail,count),
            critic_revise_recommendations=sum(r['critic']['verdict']=='REVISE' for r in completed),
            accepted=accepted_count,acceptance_rate_among_reviewed=rate(accepted_count,count),
            accepted_fraction_of_candidates=rate(accepted_count,len(rows)),
            mean_candidate_tokens=rate(sum(r['token_count'] for r in rows),len(rows)),
            completed_review_elapsed_seconds=sum(r['elapsed_seconds'] for r in completed),
            critic_issue_counts=flags,critic_issue_rates={k:rate(v,count) for k,v in flags.items()},
            actual_correction_rate=None,generator_reject_rate=None,
            recovery_scope='constructed known-bad lookup; not student-on-policy' if family=='recovery' else None)
    return dict(path=pilot.as_posix(),candidate_sha256=sha256_bytes(payload),
        reviewed_prefix_sha256=sha256_bytes(complete),reviewed_prefix_bytes=len(complete),
        incomplete_tail_bytes=len(raw)-len(complete),candidates=len(candidates),reviewed=len(reviews),
        accepted=len(accepted),accepted_with_complete_scope_pairs=len(complete_pairs),
        accepted_unknown_license=sum(not row.get('license') for row in complete_pairs),
        accepted_known_license_metadata=sum(bool(row.get('license')) for row in complete_pairs),
        # Known metadata is a rights-review input, not a new license clearance.
        candidate_license_counts=dict(Counter(row.get('license') or 'UNKNOWN' for row in candidates.values())),
        family_metrics=metrics,
        critic_config_identity_counts=dict(Counter(r.get('critic_runtime_config_sha256') or 'UNKNOWN_HISTORICAL'
                                                 for r in reviews.values())))


def inspect_coverage(pilots,pilot_config):
    config=json.loads(Path(pilot_config).read_bytes())
    reports=[inspect_pilot(p) for p in pilots]
    present={family for report in reports for family in report['family_metrics']}
    total=sum(r['candidates'] for r in reports)
    required=config['task_family_coverage']
    return dict(schema_version=1,measured_at_utc=datetime.now(timezone.utc).isoformat(),
        status='partial_pilot_coverage_not_frozen',pilots=reports,
        prepared_task_instances=total,minimum_target=config['target_task_instances']['minimum'],
        shortfall_to_minimum=max(0,config['target_task_instances']['minimum']-total),
        missing_required_task_families=[family for family in required if family not in present],
        training_ready=False,student_training_authorized=False,
        metric_limits=['Counts reflect durable completed review prefixes, not the entire running campaign.',
            'Objective/critic evidence is inspected here, not re-executed.',
            'Known license metadata does not assert complete per-file or generated-output rights clearance.',
            'REVISE recommendations are not measured successful corrections.',
            'Review elapsed times exclude unsaved failed attempts and are not pure inference timings.',
            'Source extraction and constructed recovery do not establish executable SWE or on-policy recovery coverage.'])

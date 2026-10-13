from dataclasses import asdict
import json

import pytest

from forgebuilder.pipeline import CriticResult, VerificationResult
from forgebuilder.quality import inspect_coverage, inspect_pilot
from forgebuilder.schema import sha256_bytes


def make_pilot(tmp_path,*,license=None,tail=b'',pair=False):
    rows=[{'sample_id':'sample-a','task_family':'code_from_scratch','token_count':12,'license':license},
          {'sample_id':'sample-b','task_family':'code_from_scratch','token_count':18,'license':license}]
    if pair:
        for row in rows: row['pair_id']='paired'
    payload=b''.join((json.dumps(r)+'\n').encode() for r in rows)
    (tmp_path/'candidates.jsonl').write_bytes(payload)
    (tmp_path/'preparation.json').write_text(json.dumps({'candidate_sha256':sha256_bytes(payload)}))
    review=dict(sample_id='sample-a',task_family='code_from_scratch',accepted=True,fresh_context=True,
        critic=asdict(CriticResult('PASS',.9)),elapsed_seconds=2.5,
        objective_verification=asdict(VerificationResult('PASS','fixture','v1','a'*64,'b'*64,passed=1)))
    (tmp_path/'reviews.jsonl').write_bytes((json.dumps(review)+'\n').encode()+tail)
    return review


def test_incomplete_live_review_tail_stays_pending(tmp_path):
    make_pilot(tmp_path,tail=b'{"unfinished":')
    report=inspect_pilot(tmp_path)
    assert report['reviewed']==1 and report['candidates']==2
    assert report['incomplete_tail_bytes']==len(b'{"unfinished":')
    assert report['accepted_unknown_license']==1
    metrics=report['family_metrics']['code_from_scratch']
    assert metrics['acceptance_rate_among_reviewed']==1
    assert metrics['accepted_fraction_of_candidates']==.5
    assert metrics['mean_candidate_tokens']==15
    assert metrics['actual_correction_rate'] is None


def test_incomplete_scope_pair_is_not_counted_as_exportable(tmp_path):
    make_pilot(tmp_path,license='MIT',pair=True)
    report=inspect_pilot(tmp_path)
    assert report['accepted']==1
    assert report['accepted_with_complete_scope_pairs']==0
    assert report['accepted_known_license_metadata']==0


@pytest.mark.parametrize('corruption',['identity','duplicate_review','acceptance','issue_flag'])
def test_corrupted_pilot_evidence_is_rejected(tmp_path,corruption):
    review=make_pilot(tmp_path)
    if corruption=='identity':
        with (tmp_path/'candidates.jsonl').open('ab') as f: f.write(b'\n')
    elif corruption=='duplicate_review':
        with (tmp_path/'reviews.jsonl').open('a') as f: f.write(json.dumps(review)+'\n')
    else:
        if corruption=='acceptance': review['accepted']=False
        else: review['critic']['scope_violation']=True
        (tmp_path/'reviews.jsonl').write_text(json.dumps(review)+'\n')
    with pytest.raises(ValueError): inspect_pilot(tmp_path)


def test_missing_families_and_license_metadata_do_not_freeze_dataset(tmp_path):
    pilot=tmp_path/'pilot'; pilot.mkdir(); make_pilot(pilot)
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'target_task_instances':{'minimum':5},
                                 'task_family_coverage':['code_from_scratch','agentic_swe']}))
    report=inspect_coverage([pilot],config)
    assert report['shortfall_to_minimum']==3
    assert report['missing_required_task_families']==['agentic_swe']
    assert report['training_ready'] is False

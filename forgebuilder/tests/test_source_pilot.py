from dataclasses import asdict
import json

import pytest

from forgebuilder.pilot import make_candidates, parse_critic, verify
from forgebuilder.schema import sha256_bytes
from forgebuilder.stop_scope import validate_stop_continue_pair


def candidates():
    parent = dict(source_id='repo:module.py', sample_id='ds2-'+'a'*64,
        file_path='module.py', repository='repo', content_sha256='b'*64,
        license='MIT',source_provenance={'revision':'abc'},eval_quarantine_status='clear_lexical_screen')
    units = [('first','@decorator\ndef first(x):\n    return x\n'),
             ('second','def second(y):\n    return y[0]\n')]
    return make_candidates(parent,units)


def test_objectives_masks_and_scope_pairs():
    rows = candidates()
    assert len(rows)==5
    assert all(verify(row).status=='PASS' for row in rows)
    recovery = next(r for r in rows if r['task_family']=='recovery')
    assert recovery['trajectory']['events'][0]['loss_mask']==0
    assert recovery['trajectory']['events'][0]['target_weight']==0
    scope = [r for r in rows if r['task_family']=='stop_scope']
    validate_stop_continue_pair(*scope)
    assert not any(r['execution_verified'] for r in rows)


def test_verifier_rejects_extra_tail_and_does_not_trust_names():
    rows = candidates()
    single = rows[3]
    single['candidate'] += '\ndef unauthorized_tail():\n    pass\n'
    assert verify(single).status=='FAIL'
    listing = rows[0]
    listing['task']['names'] = ['invented']
    listing['candidate'] = '{"definitions":["invented"]}'
    assert verify(listing).status=='FAIL'


def response(**overrides):
    value = dict(verdict='PASS',confidence=.9,functional_issue=False,hallucination=False,
        scope_violation=False,stop_issue=False,reasoning_issue=False,test_gaming=False,
        needs_deeper_review=False,issue_summary='')
    value.update(overrides)
    return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}


def test_incomplete_or_contradictory_review_never_passes():
    result = response()
    result['choices'][0]['finish_reason']='length'
    with pytest.raises(ValueError,match='retry review'):
        parse_critic(result,'model','version')
    with pytest.raises(ValueError,match='contradictory'):
        parse_critic(response(hallucination=True),'model','version')
    with pytest.raises(ValueError):
        parse_critic(response(confidence=100),'model','version')
    assert parse_critic(response(),'model','version').verdict=='PASS'

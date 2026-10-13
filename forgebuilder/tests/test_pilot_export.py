from dataclasses import asdict
import json

from forgebuilder.pilot import make_candidates, verify
from forgebuilder.pipeline import CriticResult
from tools.export_dataset_v2_pilot import export


def test_partial_source_export_drops_incomplete_scope_pair(tmp_path):
    parent=dict(source_id='repo:module.py',sample_id='ds2-'+'a'*64,
                file_path='module.py',repository='fixture',split='train',content_sha256='b'*64,
                license='MIT',source_provenance={'revision':'abc'},
                eval_quarantine_status='clear_lexical_screen')
    rows=make_candidates(parent,[('first','def first(x):\n    return x\n'),
                                 ('second','def second(x):\n    return x+1\n')])
    (tmp_path/'candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    selected=[rows[0],rows[3]]
    reviews=[dict(sample_id=r['sample_id'],accepted=True,
                  objective_verification=asdict(verify(r)),
                  critic=asdict(CriticResult('PASS',.9))) for r in selected]
    (tmp_path/'reviews.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reviews))
    receipt=export(tmp_path)
    assert receipt['status']=='partial_reviewed_snapshot_not_frozen'
    assert receipt['accepted_records']==1
    assert 'Source-grounded' in receipt['coverage']


def test_complete_generated_export_reports_executable_scope(tmp_path,monkeypatch):
    parent=dict(source_id='fixture',sample_id='ds2-'+'a'*64,file_path='fixture.py',
                repository='fixture',split='train',content_sha256='b'*64,license='MIT',
                source_provenance={'revision':'abc'},eval_quarantine_status='clear_lexical_screen')
    row=make_candidates(parent,[('first','def first(x):\n    return x\n'),
                                ('second','def second(x):\n    return x+1\n')])[0]
    proof=verify(row)
    row['task']={'kind':'new_project_function'}
    row['task_family']='code_from_scratch'
    monkeypatch.setattr('tools.export_dataset_v2_pilot.verify',lambda _:proof)
    (tmp_path/'candidates.jsonl').write_text(json.dumps(row)+'\n')
    review=dict(sample_id=row['sample_id'],accepted=True,
                objective_verification=asdict(proof),critic=asdict(CriticResult('PASS',.9)))
    (tmp_path/'reviews.jsonl').write_text(json.dumps(review)+'\n')
    receipt=export(tmp_path)
    assert receipt['status']=='review_complete_snapshot_not_dataset_frozen'
    assert receipt['accepted_records']==1
    assert 'WASI cases' in receipt['coverage']
    assert not (tmp_path/'accepted_snapshot_receipt.tmp').exists()

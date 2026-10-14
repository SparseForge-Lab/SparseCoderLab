import json

import pytest

from forgebuilder.terminal_tasks import locate_definitions,verify_terminal,prepare_terminal_pilot,finalize_terminal_quarantine
from forgebuilder.pilot import make_candidates
from forgebuilder.schema import canonical_json,sha256_bytes
from tokenizers import Tokenizer,models


def test_actual_terminal_search_reports_relative_lines_without_executing_source(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    source="raise RuntimeError('source must never execute')\n@decorator\ndef first(x):\n    return x\n\nasync def second(y):\n    return y\n"
    proof=locate_definitions(source)
    assert proof['objective_pass']
    assert proof['definitions']==[{'name':'first','excerpt_line':3},{'name':'second','excerpt_line':6}]
    assert proof['runtime']['candidate_code_executed'] is False
    assert proof['runtime']['shell'] is False
    assert not list((tmp_path/'.local/dataset_v2/terminal_staging').iterdir())
    record={'task':{'evidence':{'excerpt':source,'terminal_execution':proof}},
            'candidate':json.dumps({'definitions':proof['definitions']})}
    assert verify_terminal(record).status=='PASS'
    record['candidate']=json.dumps({'definitions':[{'name':'first','excerpt_line':99}]})
    assert verify_terminal(record).status=='FAIL'


def test_changed_terminal_environment_cannot_reuse_prior_execution(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    source='def first():\n    return 1\n'
    proof=locate_definitions(source)
    proof['runtime']['executable_sha256']='0'*64
    with pytest.raises(ValueError,match='identity changed'):
        verify_terminal({'task':{'evidence':{'excerpt':source,'terminal_execution':proof}},'candidate':'{}'})


def test_bounded_input_and_missing_runtime_are_rejected(monkeypatch):
    with pytest.raises(ValueError,match='bounded input'):
        locate_definitions('x'*20001)
    monkeypatch.setattr('forgebuilder.terminal_tasks.shutil.which',lambda _:None)
    with pytest.raises(RuntimeError,match='unavailable'):
        locate_definitions('def first():\n    pass\n')


def test_new_terminal_tasks_are_screened_before_any_review(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    source=tmp_path/'source'; source.mkdir()
    parent={'sample_id':'ds2-'+'a'*64,'source_id':'repo:file.py','file_path':'file.py',
            'repository':'repo','content_sha256':'b'*64,'license':'MIT',
            'source_provenance':{'revision':'fixture'},'eval_quarantine_status':'clear_lexical_screen'}
    rows=make_candidates(parent,[('first','def first():\n    return 1\n'),
                                 ('second','def second():\n    return 2\n')])
    payload=b''.join(canonical_json(r)+b'\n' for r in rows)
    (source/'candidates.jsonl').write_bytes(payload)
    negatives=[{'path':'fixture-negative.json','sha256':'c'*64}]
    (source/'preparation.json').write_text(json.dumps({'candidate_sha256':sha256_bytes(payload),
                                                     'evaluation_negative_registry':negatives}))
    tokenizer=Tokenizer(models.WordLevel({'[UNK]':0},unk_token='[UNK]'))
    output=tmp_path/'terminal'
    assert prepare_terminal_pilot(source,output,tokenizer)['candidate_count']==1
    original=(output/'candidates.jsonl').read_bytes()
    class Flagged:
        def inspect(self,text): return {'status':'flagged','matches':['fixture']}
    monkeypatch.setattr('forgebuilder.terminal_tasks.evaluation_registry',lambda _: (Flagged(),negatives))
    receipt=finalize_terminal_quarantine(source,output)
    assert receipt['candidate_count']==0 and receipt['quarantined_records']==1
    assert (output/'candidates.pre_quarantine.jsonl').read_bytes()==original
    (output/'reviews.jsonl').write_bytes(b'')
    with pytest.raises(ValueError,match='reviewed pool'): finalize_terminal_quarantine(source,output)

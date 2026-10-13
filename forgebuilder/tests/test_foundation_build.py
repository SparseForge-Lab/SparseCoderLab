import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from forgebuilder.foundation import IndexedQuarantine, build, frozen_assignments
from forgebuilder.schema import sha256_bytes


def test_reference_overlap_detected_inside_larger_file():
    text = 'the marked benchmark answer contains eight special words exactly here'
    screen = IndexedQuarantine([('eval-only', text)])
    assert screen.inspect('other unrelated words\n' + text + '\nextra content')['status'] == 'flagged'
    assert screen.inspect('def distinct_function(): return 1234')['status'] == 'clear_lexical_screen'
    assert not screen.inspect(text)['universal_contamination_claim']


def fixture_config(tmp_path):
    splits = tmp_path / 'splits'
    splits.mkdir()
    rows = []
    for repo, name, body in [('train/repo','a.py','value = 123\n'),
                             ('train/repo','b.py','other = 456\n'),
                             ('train/repo','c.py','third = 789\n'),
                             ('val/repo','z.py','held_out = 345\n')]:
        rows.append(dict(repository=repo, source_url='https://example.org/'+repo, revision='abc123',
            file_path=name, text=f'<repo>{repo}<file>{name}\n'+body, raw_sha256=sha256_bytes(body.encode()),
            normalized_sha256=sha256_bytes(body.rstrip().encode()), source_license_sha256='a'*64,
            license='MIT', git_blob='123', language='Python', repository_family=repo, role='implementation',
            kind='code', fim={'applied':False}))
    for split, repo in [('train','train/repo'),('val','val/repo')]:
        (splits / f'{split}_documents.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['repository']==repo))
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'sources':[dict(id=repo, revision='abc123', root_license_sha256='a'*64,
        planned_split=split, license_evidence=[{'path':'LICENSE','sha256':'a'*64}],review_scope='fixture')
        for repo,split in [('train/repo','train'),('val/repo','val')]]}))
    negative = tmp_path / 'eval.json'
    negative.write_text(json.dumps({'tasks':[{'prompt':'entirely separate evaluation material do not reuse this answer'}]}))
    tokenizer = Tokenizer(models.WordLevel({'[UNK]':0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    return dict(corpus=str(corpus), source_plan=str(plan), frozen_splits=str(splits),
        tokenizer=str(tmp_path/'tokenizer.json'),evaluation_negatives=[str(negative)])


def test_real_build_preserves_splits_and_verifiable_concat(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    config['evaluation_negatives'] = ['eval.json']
    output = tmp_path/'output'
    report = build(config, output)
    records = [json.loads(x) for x in (output/'foundation.jsonl').read_text().splitlines()]
    views = [json.loads(x) for x in (output/'views.jsonl').read_text().splitlines()]
    assert len(records) == 4 and len(views) == 1
    assert {r['split'] for r in records if r['repository']=='val/repo'} == {'validation'}
    originals = [json.loads(x)['text'] for x in (tmp_path/'corpus.jsonl').read_text().splitlines()][:3]
    assert views[0]['content_sha256'] == sha256_bytes('\n'.join(originals).encode())
    assert report['status'] == 'foundation_views_built_not_training_ready'
    assert not report['student_training_authorized']
    with pytest.raises(FileExistsError):
        build(config, output)


def test_frozen_split_conflict_rejected(tmp_path):
    config = fixture_config(tmp_path)
    directory = tmp_path/'splits'
    with (directory/'val_documents.jsonl').open('a') as stream:
        stream.write((directory/'train_documents.jsonl').read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError, match='both splits'):
        frozen_assignments(directory)

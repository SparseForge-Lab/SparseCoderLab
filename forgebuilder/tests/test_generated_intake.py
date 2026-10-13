import json

import pytest
from tokenizers import Tokenizer, models

from forgebuilder.generated import prepare_generated_pilot, verify_generated
from forgebuilder.schema import sha256_bytes
from tools import qualify_dataset_v2_generators as qualifier


def test_changed_task_oracle_is_rejected_before_execution(tmp_path):
    registry=tmp_path/'registry.json'
    registry.write_text('{"tasks":[]}')
    with pytest.raises(ValueError,match='registry changed'):
        verify_generated({'task':{'registry_path':str(registry),'registry_sha256':'0'*64}})


def test_changed_generated_code_cannot_reuse_old_verification(tmp_path,monkeypatch):
    foundation=tmp_path/'foundation'
    foundation.mkdir()
    tokenizer=tmp_path/'tokenizer.json'
    Tokenizer(models.WordLevel({'[UNK]':0},unk_token='[UNK]')).save(str(tokenizer))
    (foundation/'build_receipt.json').write_text(json.dumps({'build_config':{'tokenizer':str(tokenizer)},
        'evaluation_negative_registry':[]}))
    qualification=tmp_path/'qualification'
    qualification.mkdir()
    (qualification/'receipt.json').write_text(json.dumps({'model':{'repository':'fixture/model'},
        'results':[{'task_id':'segment_read_plan_v1','candidate_sha256':sha256_bytes(b'original code')}]}))
    (qualification/'segment_read_plan_v1.candidate.py').write_text('changed code')
    monkeypatch.setattr('forgebuilder.generated.evaluation_registry',lambda _: (None,[]))
    with pytest.raises(ValueError,match='differs from its saved qualification'):
        prepare_generated_pilot(tmp_path/'out',[qualification],foundation)
    assert not (tmp_path/'out').exists()


def test_cached_model_identity_requires_matching_mtime(tmp_path,monkeypatch):
    path=tmp_path/'model.gguf'
    path.write_bytes(b'GGUF fixture')
    model={'size_bytes':path.stat().st_size,'sha256':sha256_bytes(path.read_bytes())}
    receipt={'status':'verified_download','path':str(path),'size_bytes':model['size_bytes'],
             'sha256':model['sha256'],'mtime_ns':path.stat().st_mtime_ns}
    (tmp_path/'download_receipt.json').write_text(json.dumps(receipt))
    calls=[]
    monkeypatch.setattr(qualifier,'digest',lambda _:calls.append(True) or 'wrong')
    qualifier.verify_model(model,path)
    assert not calls
    receipt['mtime_ns']-=1
    (tmp_path/'download_receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='publisher identity'):
        qualifier.verify_model(model,path)
    assert calls

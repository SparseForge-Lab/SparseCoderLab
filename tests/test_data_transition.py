import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.config import load_config
from src.model import LanguageModel
from src.training.checkpoint import save_training
from src.training.engine import make_optimizer
from src.utils.hashing import sha256_file
from tools.prepare_data_transition import create_transition, transition_config, INHERITED_TOKENS


def test_transition_retains_weights_adam_rng_and_resets_only_declared_phase(tmp_path,monkeypatch):
    cfg=load_config('configs/test.yaml');cfg['training']['optimizer_decay_policy']='legacy_all'
    cfg['mtp']['enabled']=False
    model=LanguageModel(cfg);opt,scheduler=make_optimizer(model,cfg)
    x=torch.randint(0,512,(2,8));model(x,x)['loss'].backward();opt.step();scheduler.step()
    parent=tmp_path/'parent.pt';weights=tmp_path/'weights.pt'
    save_training(parent,model,opt,scheduler,dict(tokens_seen=INHERITED_TOKENS,step=12208,cursor=456,
        wall_time=100.,training_seconds=90.,config_hash='parent',data_hash='old',tokenizer_hash='same'))
    torch.save(model.state_dict(),weights)
    tokenizer=tmp_path/'tokenizer.json';tokenizer.write_text('frozen fixture')
    shards=tmp_path/'shards';shards.mkdir();(shards/'manifest.json').write_text('{}')
    entry=dict(full_resume_state=dict(path=str(parent),sha256=sha256_file(parent)),
        model_weights=dict(path=str(weights),sha256=sha256_file(weights)),tokenizer_sha256=sha256_file(tokenizer),
        config_fingerprint='parent',data_manifest_sha256='old')
    data=copy.deepcopy(cfg);data['data'].update(tokenizer=str(tokenizer),shards=str(shards))
    new=transition_config(cfg,data);new['training'].update(context=32,microbatch=256,accumulation=1)
    original=torch.load(parent,map_location='cpu',weights_only=False)
    destination=tmp_path/'transition.pt';receipt=create_transition(new,entry,destination)
    state=torch.load(destination,map_location='cpu',weights_only=False)
    assert receipt['scheduler_last_epoch']==0 and receipt['initial_group_lrs']==[new['training']['lr']/500]*2
    assert state['metadata']['step']==12208 and state['metadata']['cursor']==0
    assert state['metadata']['tokens_seen']==INHERITED_TOKENS and state['metadata']['new_phase_tokens']==0
    assert state['metadata']['data_hash']==sha256_file(shards/'manifest.json')
    assert all(torch.equal(v,state['model'][k]) for k,v in original['model'].items())
    old_ids=dict(zip((name for name,_ in model.named_parameters()),original['optimizer']['param_groups'][0]['params']))
    new_ids={name:index for group in state['optimizer']['param_groups'] for name,index in zip(group['param_names'],group['params'])}
    for name,index in old_ids.items():
        for key,value in original['optimizer']['state'].get(index,{}).items():
            assert torch.equal(value,state['optimizer']['state'][new_ids[name]][key])
    assert torch.equal(original['rng']['torch'],state['rng']['torch'])
    assert np.array_equal(original['rng']['numpy'][1],state['rng']['numpy'][1])
    assert original['rng']['python']==state['rng']['python']
    assert sha256_file(parent)==entry['full_resume_state']['sha256'] and sha256_file(weights)==entry['model_weights']['sha256']
    with pytest.raises(FileExistsError): create_transition(new,entry,destination)
    tokenizer.write_text('changed')
    with pytest.raises(ValueError,match='tokenizer'): create_transition(new,entry,tmp_path/'wrong.pt')

import copy
import json
from pathlib import Path

import pytest
import torch

from forgebuilder.comparison import validate_plan, validate_state
from src.config import fingerprint


def plan():
    return json.loads(Path('configs/dataset_v2/comparison_plan.json').read_bytes())


def state_fixture():
    model=torch.nn.Linear(2,3)
    cfg={'training':{'context':1024,'microbatch':8,'accumulation':1,'lr':.0003,'min_lr_ratio':.1}}
    names=list(dict(model.named_parameters()))
    optimizer={'param_groups':[{'params':list(range(len(names))),'param_names':names,'lr':.00003}],
               'state':{i:{'step':torch.tensor(61035.),'exp_avg':torch.zeros_like(p),
                           'exp_avg_sq':torch.zeros_like(p)} for i,p in enumerate(model.parameters())}}
    state={'model':model.state_dict(),'optimizer':optimizer,
           'scheduler':{'last_epoch':48827,'_step_count':48828,'_last_lr':[.00003]},
           'rng':{'python':(),'numpy':(),'torch':torch.zeros(1,dtype=torch.uint8),
                  'cuda':[torch.zeros(1,dtype=torch.uint8)]},'scaler':None,
           'metadata':{'tokens_seen':499998720,'step':61035,'cursor':390616,
                       'phase_steps':48827,'phase_start_step':12208,'config_hash':fingerprint(cfg)}}
    return state,cfg,model


def test_plan_requires_both_aligned_gates_and_disabled_launch():
    original=plan(); validate_plan(original)
    for field,value in [('training_enabled',True),('launch_authorized',True),('gates',original['gates'][1:])]:
        changed=copy.deepcopy(original); changed[field]=value
        with pytest.raises(ValueError): validate_plan(changed)
    changed=copy.deepcopy(original); changed['gates'][0]['additional_updates']+=1
    with pytest.raises(ValueError,match='accounting'): validate_plan(changed)


def test_full_state_validates_without_weight_updates():
    state,cfg,model=state_fixture()
    result=validate_state(state,cfg,plan(),model)
    assert result['step']==61035 and result['cursor']==390616
    assert result['optimizer_parameters']==2
    assert all(torch.count_nonzero(s['exp_avg'])==0 for s in state['optimizer']['state'].values())


@pytest.mark.parametrize('change,reason',[
    ('missing_optimizer','complete training state'),
    ('wrong_config','configuration fingerprint'),
    ('wrong_moment_shape','moment shape'),
    ('duplicate_parameter','binding mismatch'),
    ('missing_slot','parameter coverage'),
    ('scheduler_counter','scheduler counter'),
    ('scheduler_lr','learning rate'),
    ('missing_cuda_rng','random streams'),
])
def test_corrupted_resume_entries_are_rejected(change,reason):
    state,cfg,model=state_fixture()
    if change=='missing_optimizer': del state['optimizer']
    elif change=='wrong_config': cfg['training']['lr']=.001
    elif change=='wrong_moment_shape': state['optimizer']['state'][0]['exp_avg']=torch.zeros(1)
    elif change=='duplicate_parameter': state['optimizer']['param_groups'][0]['param_names'][1]='weight'
    elif change=='missing_slot': state['optimizer']['state'][99]=state['optimizer']['state'][0]
    elif change=='scheduler_counter': state['scheduler']['last_epoch']+=1
    elif change=='scheduler_lr': state['scheduler']['_last_lr']=[.1]
    elif change=='missing_cuda_rng': state['rng']['cuda']=[]
    with pytest.raises(ValueError,match=reason): validate_state(state,cfg,plan(),model)


def test_json_exponent_types_are_part_of_saved_fingerprint():
    numeric=json.loads('{"epsilon":1e-06}')
    assert fingerprint(numeric)!=fingerprint({'epsilon':'1e-06'})

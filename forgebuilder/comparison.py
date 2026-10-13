"""Read-only CPU checks for the two-arm continuation's immutable entry states."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import shutil

from src.config import fingerprint
from .strata_runtime import digest


def validate_plan(plan):
    batch=plan['effective_tokens_per_update']
    start=plan['starting_total_tokens']
    if batch!=8192 or start!=499998720 or start%batch:
        raise ValueError('unexpected entry token accounting')
    expected=[('midpoint',625000448,76294),('final',750002176,91553)]
    if len(plan['gates'])!=2:
        raise ValueError('both mandatory gates are required')
    for gate,(name,tokens,updates) in zip(plan['gates'],expected):
        if (gate['name'],gate['total_tokens'],gate['total_updates'])!=(name,tokens,updates):
            raise ValueError('mandatory gate changed')
        if tokens!=updates*batch or gate['additional_updates']!=updates-start//batch:
            raise ValueError('gate update accounting mismatch')
    if plan['gates'][0]['total_tokens']*2!=start+plan['gates'][1]['total_tokens']:
        raise ValueError('midpoint is not halfway')
    if plan['variants']!=['sparse75_ngram10m','sparse75_ngram25m']:
        raise ValueError('both retained Ngram arms are required')
    if plan['training_enabled'] or plan['launch_authorized']:
        raise ValueError('read-only preparation requires disabled launch flags')


def validate_state(state,config,plan,model):
    """Check bindings and shapes without a forward pass or optimizer update."""
    if set(state)!={'model','optimizer','scheduler','rng','scaler','metadata'}:
        raise ValueError('not a complete training state')
    meta=state['metadata']; t=config['training']
    if meta['tokens_seen']!=plan['starting_total_tokens'] or meta['step']*8192!=meta['tokens_seen']:
        raise ValueError('wrong checkpoint endpoint')
    if t['context']*t['microbatch']*t['accumulation']!=8192:
        raise ValueError('effective batch changed')
    if fingerprint(config)!=meta['config_hash']:
        raise ValueError('configuration fingerprint mismatch')
    expected=model.state_dict()
    if expected.keys()!=state['model'].keys() or any(expected[k].shape!=v.shape or expected[k].dtype!=v.dtype
                                                       for k,v in state['model'].items()):
        raise ValueError('model architecture or dtype mismatch')
    parameters=dict(model.named_parameters())
    seen_names=set(); seen_ids=set()
    for group in state['optimizer']['param_groups']:
        if len(group['params'])!=len(group.get('param_names',[])):
            raise ValueError('optimizer parameter names missing')
        for index,name in zip(group['params'],group['param_names']):
            if index in seen_ids or name in seen_names or name not in parameters:
                raise ValueError('optimizer parameter binding mismatch')
            seen_ids.add(index); seen_names.add(name)
            slot=state['optimizer']['state'][index]
            if any(slot[k].shape!=parameters[name].shape for k in ('exp_avg','exp_avg_sq')):
                raise ValueError('optimizer moment shape mismatch')
            step=float(slot['step'])
            if not math.isfinite(step) or step!=int(step) or not 0<=step<=meta['step']:
                raise ValueError('optimizer step mismatch')
    if seen_names!=set(parameters) or seen_ids!=set(state['optimizer']['state']):
        raise ValueError('optimizer parameter coverage mismatch')
    scheduler=state['scheduler']
    if scheduler['last_epoch']!=meta['phase_steps'] or meta['phase_steps']+meta['phase_start_step']!=meta['step']:
        raise ValueError('scheduler counter mismatch')
    if scheduler['_step_count']!=scheduler['last_epoch']+1:
        raise ValueError('scheduler update count mismatch')
    groups=state['optimizer']['param_groups']
    if len(scheduler['_last_lr'])!=len(groups):
        raise ValueError('scheduler group coverage mismatch')
    for lr,group in zip(scheduler['_last_lr'],groups):
        if not math.isclose(lr,group['lr'],rel_tol=1e-12) or not math.isclose(lr,t['lr']*t['min_lr_ratio'],rel_tol=1e-12):
            raise ValueError('entry learning rate differs from historical floor')
    rng=state['rng']
    if set(rng)!={'python','numpy','torch','cuda'} or not rng['cuda']:
        raise ValueError('saved CPU/CUDA random streams missing')
    if not isinstance(meta['cursor'],int) or meta['cursor']<0:
        raise ValueError('invalid saved data cursor')
    return dict(tokens_seen=meta['tokens_seen'],step=meta['step'],cursor=meta['cursor'],
                phase_steps=meta['phase_steps'],scheduler_last_epoch=scheduler['last_epoch'],
                optimizer_parameters=len(seen_names),model_tensors=len(expected),
                saved_cuda_rng_streams=len(rng['cuda']),group_lrs=scheduler['_last_lr'],
                config_fingerprint=meta['config_hash'],context_tokens=t['context'])


def inspect_entries(plan_path):
    import torch
    from src.model import LanguageModel
    from src.training.data import PackedStream
    plan_path=Path(plan_path)
    plan=json.loads(plan_path.read_bytes()); validate_plan(plan)
    rows={}; configs={}
    for variant in plan['variants']:
        entry=plan['starting_full_state_checkpoints'][variant]
        path=Path(entry['path']); config_path=path.parent.parent/'config.json'
        # JSON is intentional: YAML's interpretation of exponent literals can
        # alter types and break the original configuration fingerprint.
        config=json.loads(config_path.read_bytes()); configs[variant]=config
        if path.stat().st_size!=entry['size_bytes'] or digest(path)!=entry['sha256']:
            raise ValueError('entry checkpoint identity mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        with torch.device('meta'):
            model=LanguageModel(config)
        row=validate_state(state,config,plan,model)
        for field,actual in [('tokenizer_hash',digest(Path(config['data']['tokenizer']))),
                             ('data_hash',digest(Path(config['data']['shards'])/'manifest.json'))]:
            if state['metadata'][field]!=actual:
                raise ValueError('saved tokenizer or source dataset identity mismatch')
            row[field]=actual
        stream=PackedStream(Path(config['data']['shards']),'train',config['training']['context'],
                            config['data']['seed'],cursor=state['metadata']['cursor'])
        x,y=stream.next(config['training']['microbatch'],'cpu')
        row.update(next_source_batch_sha256=fingerprint({'x':x.tolist(),'y':y.tolist(),
                       'segments':stream.last_segment_ids.tolist() if stream.last_segment_ids is not None else None}),
                   next_cursor=stream.cursor,config_path=config_path.as_posix(),config_sha256=digest(config_path),
                   checkpoint_sha256=entry['sha256'],checkpoint_bytes=entry['size_bytes'],
                   entry_checkpoint_path=path.as_posix())
        if hasattr(stream.tokens,'close'): stream.tokens.close()
        rows[variant]=row
        del stream,state,model,x,y
        gc.collect()
    first,second=(configs[v] for v in plan['variants'])
    for section in ('model','mtp','data'):
        if first[section]!=second[section]:
            raise ValueError('shared architecture, tokenizer or source data differs across arms')
    for field in ('context','microbatch','accumulation','seed','lr','betas','weight_decay'):
        if first['training'][field]!=second['training'][field]:
            raise ValueError('shared entry training setting differs')
    if {k:v for k,v in first['memory'].items() if k!='rows'}!={k:v for k,v in second['memory'].items() if k!='rows'}:
        raise ValueError('memory settings differ beyond the intended table capacity')
    a,b=(rows[v] for v in plan['variants'])
    if a['cursor']!=b['cursor'] or a['next_source_batch_sha256']!=b['next_source_batch_sha256']:
        raise ValueError('saved source-stream position differs across arms')
    free=shutil.disk_usage(Path(plan_path).resolve().parent).free
    minimum=20*1024**3
    return dict(schema_version=1,status='entry_structure_verified_launch_blocked',
                plan_sha256=digest(plan_path),entry_checks_passed=True,variants=rows,
                storage=dict(measured_free_bytes=free,minimum_reserve_bytes=minimum,
                             current_reserve_passed=free>=minimum,
                             limit='Current free space only; final dataset/checkpoint peak budget remains to be frozen.'),
                training_launched=False,gpu_used=False,
                open_launch_gates=['licensed_complete_dataset_and_quarantine','shared_frozen_packing_recipe_and_order',
                                  'explicit_dataset_transition_scheduler_and_cursor_policy',
                                  'numerical_interrupted_vs_continuous_resume','final_peak_storage_budget',
                                  'explicit_student_training_authorization'],
                scope='CPU memory-mapped full-state identity, configuration, shape, optimizer binding, scheduler and historical source-cursor inspection. No weight update, CUDA RNG restoration, numerical resume equivalence or Dataset V2 transition was executed.')

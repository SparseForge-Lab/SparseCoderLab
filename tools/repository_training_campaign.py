"""Sequential frozen-corpus training using the unchanged validated engine.

The wrapper adds identity/failure checks, verified two-generation rolling states,
and telemetry. It does not change optimizer updates, data ordering or RNG.
"""
from __future__ import annotations
import argparse, gc, json, math, os, shutil, time, traceback
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import psutil
import torch
from src.config import fingerprint
from src.training import engine
from src.training.checkpoint import atomic_save
from src.utils.hashing import sha256_file
from tools.generate_canonical_python import gpu_allowed
from tools.transition_cuda_preflight import finite_tensors

TAGS=('dense75_ref','sparse75','sparse75_ngram10m','sparse75_ngram25m')
INHERITED=100007936
START_STEP=12208
TARGET=250003456

def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    os.replace(temporary,path)

def check_metadata(meta, cfg, scheduler):
    phase=meta['step']-START_STEP
    if not 0<=phase<=18310: raise ValueError('Phase step outside frozen target')
    expected=dict(inherited_tokens=INHERITED,phase_start_step=START_STEP,phase_steps=phase,
        tokens_seen=INHERITED+phase*8192,new_phase_tokens=phase*8192,cursor=phase*8,
        config_hash=fingerprint(cfg))
    for key,value in expected.items():
        if meta.get(key)!=value: raise ValueError(f'Checkpoint identity/count mismatch: {key}')
    if scheduler['last_epoch']!=phase: raise ValueError('Phase scheduler counter mismatch')

def rate(cfg, phase):
    t=cfg['training']
    if phase<t['warmup_steps']: return t['lr']*(phase+1)/t['warmup_steps']
    fraction=min((phase-t['warmup_steps'])/(t['schedule_steps']-t['warmup_steps']),1)
    return t['lr']*(t['min_lr_ratio']+(1-t['min_lr_ratio'])*(1+math.cos(math.pi*fraction))/2)

def verify_state(path, cfg):
    state=torch.load(path,map_location='cpu',weights_only=False)
    check_metadata(state['metadata'],cfg,state['scheduler'])
    expected=rate(cfg,state['scheduler']['last_epoch'])
    if not all(math.isclose(group['lr'],expected,rel_tol=1e-12) for group in state['optimizer']['param_groups']):
        raise ValueError('Adam group LR does not match frozen phase schedule')
    if len(state['optimizer']['param_groups'])!=2 or not state['optimizer']['state']:
        raise ValueError('Missing inherited Adam state/groups')
    if set(state['rng'])!={'python','numpy','torch','cuda'} or not state['rng']['cuda']:
        raise ValueError('Missing required RNG streams')
    finite_tensors(state['model']);finite_tensors(state['optimizer'])
    meta=dict(state['metadata']);del state;gc.collect()
    return dict(path=path.as_posix(),sha256=sha256_file(path),size_bytes=path.stat().st_size,
        step=meta['step'],tokens_seen=meta['tokens_seen'],new_phase_tokens=meta['new_phase_tokens'],
        cursor=meta['cursor'],phase_steps=meta['phase_steps'],finite_model_and_adam=True,
        scheduler_cursor_counts_verified=True,rng_streams_present=True)

def check_identities(identities):
    for name,digest in identities.items():
        if Path(name).name.casefold()=='handoff.md': raise ValueError('Consumed file excluded')
        if sha256_file(Path(name))!=digest: raise ValueError('Run identity changed: '+name)

def run_variant(tag, run_root, entry, *, max_steps=None):
    run_dir=run_root/tag
    cfg=json.loads((run_dir/'config.json').read_text(encoding='utf8'))
    initial=entry['initial_copies'][tag]
    if fingerprint(cfg)!=initial['config_hash']: raise ValueError('Transition configuration changed')
    if sha256_file(run_dir/'checkpoints/initial.pt')!=initial['checkpoint_sha256']:
        raise ValueError('Immutable initial copy changed')
    check_identities(entry['operative_sources_verified'])
    identities=dict(entry['operative_sources_verified'])
    for directory in ('src',):
        identities.update({path.as_posix():sha256_file(path) for path in Path(directory).rglob('*.py')})
    identities['tools/repository_training_campaign.py']=sha256_file(Path(__file__))
    identities[cfg['data']['tokenizer']]=sha256_file(Path(cfg['data']['tokenizer']))
    for path in Path(cfg['data']['shards']).iterdir():
        if path.is_file(): identities[path.as_posix()]=sha256_file(path)
    index=Path('results/research_v2_real')/f"evaluation_index_{cfg['data']['dataset_version']}.json"
    identities[index.as_posix()]=sha256_file(index)
    check_identities(identities)
    control=run_dir/'checkpoint_verification.json'
    receipt=json.loads(control.read_text()) if control.exists() else {}
    resume=run_dir/'checkpoints/initial.pt'
    candidates=sorted((row for row in receipt.values() if isinstance(row,dict) and 'step' in row),key=lambda row:row['step'],reverse=True)
    for row in candidates:
        path=Path(row['path'])
        if path.exists() and sha256_file(path)==row['sha256']:
            verify_state(path,cfg);resume=path;break
    if candidates and resume.name=='initial.pt': raise ValueError('No verified rolling checkpoint; refuse silent phase restart')
    resumed=verify_state(resume,cfg)
    baseline=json.loads(Path('results/research_v2_real/transition_cuda_preflight_r2.json').read_text())['variants'][tag]
    expected_peak=baseline['peak_allocated_bytes']
    log=run_dir/'campaign_events.jsonl'
    started=time.perf_counter()
    original_save,original_make,original_validate,original_util=engine.save_training,engine.make_optimizer,engine.validate,engine.read_utilization
    holder={};collapse={};starvation={};cpu_peak=[0]
    def event(value):
        with log.open('a',encoding='utf8') as handle:
            handle.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),variant=tag,**value))+'\n')
    def save(path, model, optimizer, scheduler, meta, limit=None):
        gpu_allowed();check_identities(identities);check_metadata(meta,cfg,scheduler.state_dict())
        if fingerprint(json.loads((run_dir/'config.json').read_text()))!=fingerprint(cfg):
            raise ValueError('Resolved run configuration changed')
        if shutil.disk_usage(run_dir).free<20*1024**3+initial['checkpoint_bytes']*2:
            raise RuntimeError('Rolling checkpoint would breach 20GiB reserve')
        began=time.perf_counter()
        prior=receipt.get('last')
        if prior and prior['step']!=meta['step']:
            if sha256_file(path)!=prior['sha256']: raise ValueError('Current rolling checkpoint corrupted')
            previous=path.with_name('previous.pt');temp=previous.with_suffix('.tmp')
            shutil.copy2(path,temp);os.replace(temp,previous)
            receipt['previous']=dict(prior,path=previous.as_posix())
            write_json(control,receipt)
        original_save(path,model,optimizer,scheduler,meta,limit)
        write_seconds=time.perf_counter()-began
        checked=verify_state(path,cfg)
        receipt['last']=checked
        write_json(control,receipt)
        event(dict(stage='checkpoint_verified',**checked,write_and_rotation_seconds=write_seconds,
            verification_seconds=time.perf_counter()-began-write_seconds,free_bytes=shutil.disk_usage(run_dir).free))
        print(json.dumps(dict(stage='checkpoint_verified',variant=tag,tokens=meta['tokens_seen'],phase_updates=meta['phase_steps'])),flush=True)
    def make(model, config):
        optimizer,scheduler=original_make(model,config)
        holder.update(model=model,optimizer=optimizer,scheduler=scheduler)
        def guarded_step(optimizer,args,kwargs):
            gpu_allowed()
            for group in optimizer.param_groups:
                if not math.isclose(group['lr'],rate(cfg,scheduler.last_epoch),rel_tol=1e-12):
                    raise ValueError('LR changed outside the frozen scheduler')
        optimizer.register_step_pre_hook(guarded_step)
        # Read-only hook; no tensor modification, graph retention or random draws.
        def aux_hook(module,args,output):
            if module.training and (START_STEP+scheduler.last_epoch+1)%50==0:
                holder['aux_loss']=float(output['aux_loss'].detach())
        model.register_forward_hook(aux_hook)
        return optimizer,scheduler
    def utilization():
        model=holder['model'];scheduler=holder['scheduler']
        peak=torch.cuda.max_memory_allocated()
        if peak>max(expected_peak*2,expected_peak+1024**3):
            raise RuntimeError('Allocated VRAM materially exceeds validated preflight')
        routes={}
        for layer,value in model.routes().items():
            loads=value['load'].cpu();freq=loads/loads.sum();n=len(loads)
            maximum=float(freq.max());entropy=float(value['entropy'])
            collapse[layer]=collapse.get(layer,0)+1 if maximum>.98 and entropy<.05*math.log(n) else 0
            starvation[layer]=starvation.get(layer,0)+1 if int((loads==0).sum())>=n-1 else 0
            if collapse[layer]>=10 or starvation[layer]>=10:
                raise RuntimeError(f'Sustained severe router collapse/starvation in layer {layer}')
            routes[layer]=dict(entropy=entropy,frequency=freq.tolist(),dead_this_sample=int((loads==0).sum()),
                under_one_percent_this_sample=int((freq<.01).sum()),maximum_frequency=maximum)
        rss=psutil.Process().memory_info().rss;cpu_peak[0]=max(cpu_peak[0],rss)
        event(dict(stage='training_sample',step=START_STEP+scheduler.last_epoch,phase_steps=scheduler.last_epoch,
            lr=scheduler.get_last_lr(),cpu_rss_bytes=rss,cpu_rss_peak_sampled_bytes=cpu_peak[0],
            aux_loss=holder.get('aux_loss'),router=routes,peak_allocated_bytes=peak,
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        return original_util()
    def validate(model, config):
        began=time.perf_counter();value=original_validate(model,config)
        event(dict(stage='quick_validation',phase_steps=holder['scheduler'].last_epoch,
            seconds=time.perf_counter()-began,**value))
        if not math.isfinite(value['val_loss']) or value['val_loss']>baseline['baseline_evaluation']['val_loss']+1.5:
            raise RuntimeError('Material held-out loss regression; diagnose before further training')
        return value
    engine.save_training=save;engine.make_optimizer=make;engine.read_utilization=utilization;engine.validate=validate
    event(dict(stage='begin',resume=resumed,identities=identities,deterministic_algorithms=True))
    try:
        result=engine.run(cfg,run_dir,resume=resume,max_steps=max_steps)
        result['campaign_process_seconds']=time.perf_counter()-started
        result['cpu_rss_peak_sampled_bytes']=cpu_peak[0]
        result['train_updates_per_second']=result['train_step_tok_s']/8192
        result['final_checkpoint_verified']=receipt['last']
        if result['tokens_seen']==TARGET:
            full=run_dir/'checkpoints/latest_verified_250M.pt'
            model_bytes=sum(value.numel()*value.element_size() for value in holder['model'].state_dict().values())
            if shutil.disk_usage(run_dir).free<20*1024**3+initial['checkpoint_bytes']+model_bytes+64*1024**2:
                raise RuntimeError('Final promotion would breach 20GiB reserve')
            temporary=full.with_suffix('.tmp');shutil.copy2(run_dir/'checkpoints/last.pt',temporary);os.replace(temporary,full)
            state=torch.load(full,map_location='cpu',weights_only=False)
            weights=run_dir/'checkpoints/model_250M.pt'
            atomic_save(weights,state['model']);del state;gc.collect()
            result['promoted_full']=verify_state(full,cfg)
            result['model_only']=dict(path=weights.as_posix(),sha256=sha256_file(weights),size_bytes=weights.stat().st_size)
        write_json(run_dir/'campaign_summary.json',result)
        event(dict(stage='end',tokens_seen=result['tokens_seen'],process_seconds=result['campaign_process_seconds']))
        return result
    except BaseException as error:
        failure=dict(stage='failure',error=repr(error),traceback=traceback.format_exc(),seconds=time.perf_counter()-started,
            latest_verified=receipt)
        write_json(run_dir/f"failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",failure)
        event(failure);raise
    finally:
        engine.save_training=original_save;engine.make_optimizer=original_make;engine.validate=original_validate;engine.read_utilization=original_util
        holder.clear();gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,default=Path('experiments/prompt5_v1'))
    parser.add_argument('--variants',nargs='+',choices=TAGS,default=list(TAGS));parser.add_argument('--max-steps',type=int)
    args=parser.parse_args()
    gpu_allowed();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    entry=json.loads(Path('results/repository_training_v1/entry_verification.json').read_text())
    lock=Path('.local/process_controls/repository_campaign.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    if lock.exists():
        prior=json.loads(lock.read_text())
        if psutil.pid_exists(prior['pid']): raise RuntimeError('A repository training campaign is already running')
        lock.unlink()
    with lock.open('x') as handle: json.dump(dict(pid=os.getpid(),created_utc=datetime.now(timezone.utc).isoformat()),handle)
    try:
        for tag in args.variants:
            summary=args.run_root/tag/'campaign_summary.json'
            if summary.exists() and json.loads(summary.read_text())['tokens_seen']==TARGET:
                cfg=json.loads((args.run_root/tag/'config.json').read_text());verify_state(args.run_root/tag/'checkpoints/latest_verified_250M.pt',cfg)
                continue
            print(json.dumps(dict(stage='begin_training',variant=tag)),flush=True)
            result=run_variant(tag,args.run_root,entry,max_steps=args.max_steps)
            print(json.dumps(dict(stage='training_returned',variant=tag,tokens=result['tokens_seen'])),flush=True)
            if args.max_steps is None and result['tokens_seen']!=TARGET: raise RuntimeError('Run interrupted below common gate; resume latest verified state')
    finally: lock.unlink(missing_ok=True)

if __name__=='__main__':main()

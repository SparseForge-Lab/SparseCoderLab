"""Final-corpus CUDA smoke, frozen evaluation and exact interrupted resume."""
from __future__ import annotations
import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from src.eval.research import evaluate,freeze_evaluation
from src.model import LanguageModel
from src.training.checkpoint import save_training,resume_training,rng_state
from src.training.data import PackedStream
from src.training.engine import make_optimizer,amp,run as engine_run
from src.utils.hashing import sha256_file
from tools.generate_canonical_python import gpu_allowed
from tools.shard_data import verify_shards


def cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return value


def exact(a,b,path='state'):
    if torch.is_tensor(a):good=torch.equal(a,b)
    elif isinstance(a,np.ndarray):good=np.array_equal(a,b)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError(f'{path} keys differ')
        for k in a:exact(a[k],b[k],path+'/'+str(k))
        return
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise ValueError(f'{path} length differs')
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
        return
    else:good=a==b
    if not good:raise ValueError(f'Interrupted resume differs at {path}')


def finite_tensors(value,path='state'):
    if torch.is_tensor(value):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite tensor at {path}')
    elif isinstance(value,dict):
        for key,item in value.items():finite_tensors(item,path+'/'+str(key))
    elif isinstance(value,(tuple,list)):
        for index,item in enumerate(value):finite_tensors(item,path+'/'+str(index))


def update(model,opt,schedule,stream,cfg,meta):
    gpu_allowed();torch.cuda.synchronize();begin=time.perf_counter();opt.zero_grad(set_to_none=True)
    x,y=stream.next(cfg['training']['microbatch'],'cuda');wait=time.perf_counter()-begin
    with amp(cfg):out=model(x,y,return_outputs=False,segment_ids=stream.last_segment_ids)
    if not torch.isfinite(out['loss']):raise FloatingPointError('Nonfinite loss')
    loss=float(out['lm_loss'].detach());out['loss'].backward()
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['grad_clip'],error_if_nonfinite=True))
    del out
    opt_start=torch.cuda.Event(enable_timing=True);opt_end=torch.cuda.Event(enable_timing=True)
    opt_start.record();opt.step();opt_end.record();schedule.step();torch.cuda.synchronize()
    meta['step']+=1;meta['tokens_seen']+=x.numel();meta['cursor']=stream.cursor
    meta['new_phase_tokens']=meta['tokens_seen']-meta['inherited_tokens']
    meta['phase_steps']=meta['step']-meta['phase_start_step']
    return dict(loss=loss,grad_norm=norm,seconds=time.perf_counter()-begin,data_wait_wall_seconds=wait,
        data_wait_includes_host_to_device=True,optimizer_cuda_seconds=opt_start.elapsed_time(opt_end)/1000,
        predictions=stream.last_prediction_count,lr=schedule.get_last_lr())


def variant(tag,root,attempt='r1'):
    run=root/tag;cfg=json.loads((run/'config.json').read_text(encoding='utf8'))
    initial=run/'checkpoints'/'initial.pt';initial_hash=sha256_file(initial)
    verify_shards(cfg)
    # Exercise the actual training engine too, with no shared leaderboard writes.
    # Its single disposable update must match the custom uninterrupted path.
    if cfg['training'].get('publish_leaderboard') is not False:
        raise ValueError('Preflight must not append to the shared leaderboard')
    engine_dir=run/('engine_smoke' if attempt=='r1' else 'engine_smoke_'+attempt)
    if engine_dir.exists():raise FileExistsError('Choose new preflight outputs')
    engine_result=engine_run(cfg,engine_dir,max_steps=1,max_wall_minutes=5,resume=initial)
    engine_state=torch.load(engine_dir/'checkpoints'/'last.pt',map_location='cpu',weights_only=False)
    model=LanguageModel(cfg).cuda();opt,schedule=make_optimizer(model,cfg)
    meta=resume_training(initial,model,opt,schedule,'cuda');meta.pop('checkpoint_load_seconds',None)
    evaluation_start=time.perf_counter();index=freeze_evaluation(cfg)
    baseline=evaluate(model,cfg,index,full=True,routes=True)
    if model.memory:
        model.memory.ablate=True;baseline['residual_off']=evaluate(model,cfg,index,full=True,routes=False);model.memory.ablate=False
    evaluation_seconds=time.perf_counter()-evaluation_start
    # Evaluation does not move the training cursor. Restore inherited RNG after
    # evaluation so both uninterrupted and interrupted paths begin identically.
    meta=resume_training(initial,model,opt,schedule,'cuda');meta.pop('checkpoint_load_seconds',None)
    stream=PackedStream(Path(cfg['data']['shards']),'train',cfg['training']['context'],cfg['data']['seed'])
    torch.cuda.reset_peak_memory_stats();warmup=[]
    for index in range(3):
        warmup.append(update(model,opt,schedule,stream,cfg,meta))
        if index==0:
            exact(cpu_copy(model.state_dict()),engine_state['model'],'engine/model')
            exact(cpu_copy(opt.state_dict()),engine_state['optimizer'],'engine/optimizer')
            exact(schedule.state_dict(),engine_state['scheduler'],'engine/scheduler')
            exact(rng_state(),engine_state['rng'],'engine/rng')
    del engine_state
    checkpoint=run/'checkpoints'/('smoke_resume.pt' if attempt=='r1' else 'smoke_resume_'+attempt+'.pt');begin=time.perf_counter()
    if checkpoint.exists():raise FileExistsError('Preflight checkpoint already exists')
    save_training(checkpoint,model,opt,schedule,meta);torch.cuda.synchronize();save_seconds=time.perf_counter()-begin
    measured=[update(model,opt,schedule,stream,cfg,meta) for _ in range(6)]
    expected=cpu_copy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),rng=rng_state(),metadata=meta))
    peak_allocated=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
    del model,opt,schedule;gc.collect();torch.cuda.empty_cache()
    model=LanguageModel(cfg).cuda();opt,schedule=make_optimizer(model,cfg)
    meta=resume_training(checkpoint,model,opt,schedule,'cuda');meta.pop('checkpoint_load_seconds',None)
    stream=PackedStream(Path(cfg['data']['shards']),'train',cfg['training']['context'],cfg['data']['seed'],cursor=meta['cursor'])
    replay=[update(model,opt,schedule,stream,cfg,meta) for _ in range(6)]
    actual=cpu_copy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),rng=rng_state(),metadata=meta))
    exact(expected,actual)
    finite_tensors(expected)
    if sha256_file(initial)!=initial_hash:raise ValueError('Initial transition state changed')
    if not np.isfinite(baseline['val_loss']):raise FloatingPointError('Nonfinite evaluation')
    evaluations=[baseline]+([baseline['residual_off']] if 'residual_off' in baseline else [])
    if any(not np.isfinite(row['nll']) for evaluation in evaluations for row in evaluation['language'].values() if row['nll'] is not None):
        raise FloatingPointError('Nonfinite held-out stratum')
    result=dict(passed=True,attempt=attempt,exact_resume=True,finite_training_state=True,initial_checkpoint_sha256=initial_hash,
        actual_training_engine_smoke=dict(updates=engine_result['measured_steps'],
            new_phase_tokens=engine_result['new_phase_tokens'],total_tokens=engine_result['tokens_seen'],
            checkpoint_sha256=sha256_file(engine_dir/'checkpoints'/'last.pt'),
            first_update_exact_vs_uninterrupted=True,tokens_per_second=engine_result['train_step_tok_s']),
        smoke_checkpoint_sha256=sha256_file(checkpoint),baseline_evaluation=baseline,
        frozen_evaluation_index_sha256=sha256_file(Path('results/research_v2_real')/f"evaluation_index_{cfg['data']['dataset_version']}.json"),
        evaluation_seconds=evaluation_seconds,checkpoint_save_seconds=save_seconds,warmup_updates=warmup,
        measured_updates=measured,replayed_updates=replay,peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,
        measured_tokens_per_second=6*8192/sum(r['seconds'] for r in measured),
        new_phase_tokens=meta['new_phase_tokens'],total_tokens=meta['tokens_seen'],
        scope='9 optimizer updates on a disposable copy, 3warmup+6measured; full frozen held-out evaluation before updates. No campaign or promoted weights.')
    del model,opt,schedule,actual,expected;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--attempt',default='r1')
    args=parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_-]{0,31}',args.attempt):raise ValueError('Invalid preflight attempt name')
    if args.report.exists():raise FileExistsError('Use a new preflight report')
    gpu_allowed();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    rows={}
    for tag in ('dense75_ref','sparse75','sparse75_ngram10m','sparse75_ngram25m'):
        print(json.dumps(dict(stage='begin',variant=tag)),flush=True)
        rows[tag]=variant(tag,args.run_root,args.attempt)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.with_suffix('.partial.json').write_text(json.dumps(dict(variants=rows),indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(variant=tag,passed=True,tokens_per_second=rows[tag]['measured_tokens_per_second'])),flush=True)
    args.report.write_text(json.dumps(dict(schema_version=1,passed=True,attempt=args.attempt,variants=rows,
        script_sha256=sha256_file(Path(__file__)),device=torch.cuda.get_device_name(),
        deterministic_algorithms=True,bf16=True,tf32=True),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()

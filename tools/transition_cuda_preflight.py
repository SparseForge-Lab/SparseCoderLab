"""Final-corpus CUDA smoke, frozen evaluation and exact interrupted resume."""
from __future__ import annotations
import argparse
import gc
import json
import os
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from src.eval.research import evaluate,freeze_evaluation
from src.model import LanguageModel
from src.training.checkpoint import save_training,resume_training,rng_state
from src.training.data import PackedStream
from src.training.engine import make_optimizer,amp
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


def update(model,opt,schedule,stream,cfg,meta):
    gpu_allowed();torch.cuda.synchronize();begin=time.perf_counter();opt.zero_grad(set_to_none=True)
    x,y=stream.next(cfg['training']['microbatch'],'cuda');wait=time.perf_counter()-begin
    with amp(cfg):out=model(x,y,return_outputs=False,segment_ids=stream.last_segment_ids)
    if not torch.isfinite(out['loss']):raise FloatingPointError('Nonfinite loss')
    loss=float(out['lm_loss']);out['loss'].backward()
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


def variant(tag,root):
    run=root/tag;cfg=json.loads((run/'config.json').read_text(encoding='utf8'))
    initial=run/'checkpoints'/'initial.pt';initial_hash=sha256_file(initial)
    verify_shards(cfg)
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
    torch.cuda.reset_peak_memory_stats();warmup=[update(model,opt,schedule,stream,cfg,meta) for _ in range(3)]
    checkpoint=run/'checkpoints'/'smoke_resume.pt';begin=time.perf_counter()
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
    if sha256_file(initial)!=initial_hash:raise ValueError('Initial transition state changed')
    if not np.isfinite(baseline['val_loss']):raise FloatingPointError('Nonfinite evaluation')
    result=dict(passed=True,exact_resume=True,initial_checkpoint_sha256=initial_hash,
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
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError('Use a new preflight report')
    gpu_allowed();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    rows={}
    for tag in ('dense75_ref','sparse75','sparse75_ngram10m','sparse75_ngram25m'):
        print(json.dumps(dict(stage='begin',variant=tag)),flush=True)
        rows[tag]=variant(tag,args.run_root)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.with_suffix('.partial.json').write_text(json.dumps(dict(variants=rows),indent=2)+'\n',encoding='utf8')
        print(json.dumps(dict(variant=tag,passed=True,tokens_per_second=rows[tag]['measured_tokens_per_second'])),flush=True)
    args.report.write_text(json.dumps(dict(schema_version=1,passed=True,variants=rows,
        script_sha256=sha256_file(Path(__file__)),device=torch.cuda.get_device_name(),
        deterministic_algorithms=True,bf16=True,tf32=True),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()

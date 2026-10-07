"""Compare real microbatches at a fixed effective batch; never changes a run."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from src.config import load_config
from src.model import LanguageModel
from src.training.data import PackedStream
from src.training.engine import amp, make_optimizer, require_cuda, seed_all
from src.training.research import training_hash

OUTPUT = Path('results/research_batch_tuning')
PAIRS = ((1,8),(2,4),(4,2),(8,1))


def active_training() -> bool:
    if sys.platform != 'win32':
        raise RuntimeError('Process guard currently supports this Windows project only')
    query = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match '-m tools\\.research_train( |$)' } | "
             "Select-Object -ExpandProperty ProcessId")
    return bool(subprocess.check_output(['powershell.exe','-NoProfile','-Command',query],text=True).strip())


def one(tag: str, microbatch: int, accumulation: int, steps: int, repetition: int) -> dict:
    cfg = copy.deepcopy(load_config(f'configs/research_v1/{tag}.yaml'))
    cfg['training'].update(microbatch=microbatch,accumulation=accumulation)
    seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = cfg['training']['tf32']
    torch.backends.cudnn.allow_tf32 = cfg['training']['tf32']
    model = LanguageModel(cfg).cuda()
    opt,scheduler = make_optimizer(model,cfg)
    stream = PackedStream(Path(cfg['data']['shards']),'train',cfg['training']['context'],42)
    total_batch_prepare = total_optimizer = 0.0
    losses = []

    def update():
        nonlocal total_batch_prepare,total_optimizer
        opt.zero_grad(set_to_none=True)
        value = 0.0
        for _ in range(accumulation):
            begin = time.perf_counter()
            x,y = stream.next(microbatch,'cuda')
            total_batch_prepare += time.perf_counter()-begin
            with amp(cfg):
                out = model(x,y)
                loss = out['loss']/accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite benchmark loss')
            loss.backward()
            value += float(out['lm_loss'].detach())/accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['grad_clip'],error_if_nonfinite=True)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        opt.step();scheduler.step();torch.cuda.synchronize()
        total_optimizer += time.perf_counter()-begin
        return value

    try:
        for _ in range(3):
            update()
        total_batch_prepare = total_optimizer = 0.0
        torch.cuda.reset_peak_memory_stats()
        cpu_start = time.process_time()
        torch.cuda.synchronize();begin = time.perf_counter()
        for _ in range(steps):
            losses.append(update())
        torch.cuda.synchronize();wall = time.perf_counter()-begin
        cpu = time.process_time()-cpu_start
        tokens = steps*microbatch*accumulation*cfg['training']['context']
        return {'status':'measured','model':tag,'repetition':repetition,'microbatch':microbatch,'accumulation':accumulation,
                'effective_sequences':microbatch*accumulation,'context':cfg['training']['context'],'measured_steps':steps,
                'wall_seconds':wall,'wall_tokens_per_second':tokens/wall,'mean_step_seconds':wall/steps,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'optimizer_seconds':total_optimizer,'host_batch_prepare_seconds':total_batch_prepare,
                'process_cpu_seconds':cpu,'cpu_core_equivalents':cpu/wall,'first_loss':losses[0],'last_loss':losses[-1],
                'scope':'Fresh speed-only models/Adam,3 warmup updates then wall loop including data transfers and synchronization. No checkpoint or primary quality changes. Host batch time is not an isolated device stall measurement.'}
    finally:
        del model,opt,scheduler
        gc.collect();torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Execute only when training process guard is clear')
    parser.add_argument('--steps',type=int,default=20)
    args = parser.parse_args()
    assert args.steps >= 10
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if not args.run:
        plan = {'status':'queued_not_measured','models':['sparse','memory'],'candidates':[{'microbatch':m,'accumulation':a} for m,a in PAIRS],
                'effective_sequences':8,'context':1024,'steps_per_window':args.steps,'repetitions':2,
                'selection':'Compare actual repeated wall tokens/sec and memory headroom; do not assume the largest batch wins.',
                'quality_limit':'MoE auxiliary balance is microbatch-dependent. Fixed effective tokens alone does not prove identical optimization or a fair changed-policy seed replication.',
                'scheduling':'Benchmark only after any active training job ends; compare settings before selecting a training protocol.',
                'training_source_hash':training_hash()}
        (OUTPUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        print(json.dumps(plan));return
    if active_training():
        raise RuntimeError('Training is live; batch benchmark deferred without interrupting it')
    require_cuda('cuda')
    rows = []
    for repetition in (1,2):
        for tag in ('sparse','memory'):
            for microbatch,accumulation in (PAIRS if repetition==1 else tuple(reversed(PAIRS))):
                try:
                    row = one(tag,microbatch,accumulation,args.steps,repetition)
                except torch.OutOfMemoryError:
                    gc.collect();torch.cuda.empty_cache()
                    row = {'status':'oom','model':tag,'microbatch':microbatch,'accumulation':accumulation,'repetition':repetition}
                rows.append(row)
                (OUTPUT/'measurements.json').write_text(json.dumps({'status':'in_progress','rows':rows},indent=2),encoding='utf-8')
                print(json.dumps(row),flush=True)
    (OUTPUT/'measurements.json').write_text(json.dumps({'status':'complete','rows':rows,'quality_comparison_validated':False},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()

"""Bounded before/after implementation checks; canonical weights stay read-only."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

BASELINE='06f0ef1f5f9cf4ce58e9073f1ff845c8ac27df1d'
VARIANTS=('dense75_ref','sparse75','sparse75_ngram10m','sparse75_ngram25m')


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def worker(args):
    sys.path.insert(0,str(Path(args.source_root).resolve()))
    from src.config import load_config
    from src.model import LanguageModel
    from src.training.engine import make_optimizer
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU fallback')
    torch.set_num_threads(2); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=True
    cfg=load_config(f'configs/prompt3/{args.variant}.yaml')
    if args.moe_only:
        moe_worker(args,cfg);return
    cfg['training']['loss_chunk_tokens']=args.chunk
    if args.native:cfg['model']['gqa_backend']='native'
    manifest=json.loads(Path('results/prompt3/transition_checkpoint_manifest.json').read_text())
    entry=manifest['canonical_transition_checkpoints'][args.variant]['model_weights']
    initial_hash=hash_file(entry['path'])
    if initial_hash!=entry['sha256']:raise ValueError('Canonical weights hash mismatch')
    model=LanguageModel(cfg).cuda().train()
    model.load_state_dict(torch.load(entry['path'],map_location='cpu',weights_only=True))
    opt,scheduler=make_optimizer(model,cfg)
    tiny=torch.randint(0,cfg['model']['vocab_size'],(1,32),device='cuda'); target=torch.roll(tiny,-1,1)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(tiny,target)
    snapshot={'logits':out['logits'].cpu(),'routes':{i:r['experts'].cpu() for i,r in model.routes().items()}}
    del out
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(tiny,target,return_outputs=False) if args.mode=='after' else model(tiny,target)
    snapshot['loss']=float(out['loss'].detach()); out['loss'].backward()
    snapshot['gradients']={n:p.grad.detach().cpu() if p.grad is not None else None for n,p in model.named_parameters()}
    destination=Path(args.worker_output); destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(snapshot,destination.with_suffix('.pt'));del snapshot,out
    opt.zero_grad(set_to_none=True)
    x=torch.randint(0,cfg['model']['vocab_size'],(args.batch,args.context),device='cuda'); y=torch.roll(x,-1,1)
    optimizer_ms=[]
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=model(x,y,return_outputs=False) if args.mode=='after' else model(x,y)
            loss=result['loss']
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite benchmark loss')
        loss.backward(); loss_value=float(loss.detach()); del result,loss
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record(); opt.step(); end.record();scheduler.step();torch.cuda.synchronize()
        optimizer_ms.append(start.elapsed_time(end));return loss_value
    for _ in range(3):step()
    optimizer_ms.clear();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    times=[];allocations=[];reservations=[];losses=[]
    for _ in range(args.steps):
        start=time.perf_counter();losses.append(step());times.append(time.perf_counter()-start)
        allocations.append(torch.cuda.memory_allocated());reservations.append(torch.cuda.memory_reserved())
    report=dict(mode=args.mode,variant=args.variant,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
                context=args.context,microbatch=args.batch,accumulation=1,bf16=True,tf32=True,
                loss_chunk_tokens=args.chunk if args.mode=='after' else None,native_gqa=args.native,
                warmup_steps=3,measured_steps=args.steps,tokens_per_step=x.numel(),
                wall_tokens_per_second=x.numel()*args.steps/sum(times),median_step_seconds=statistics.median(times),
                optimizer_cuda_ms_median=statistics.median(optimizer_ms),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                steady_allocated_bytes=allocations,steady_reserved_bytes=reservations,losses=losses,
                canonical_weights_sha256=initial_hash,canonical_file_unchanged=hash_file(entry['path'])==initial_hash,
                decay_policy='matrix_except_ngram_v1' if args.mode=='after' else 'legacy_all',
                scope='Three warmup and bounded in-memory optimizer steps on copies of canonical weights, fixed synthetic token batches. No research run, no canonical writes. Wall throughput includes backward/clip/optimizer/scheduler/synchronize, excludes data loading, checkpoint and initialization. Short steady allocation trace cannot establish absence of every leak.')
    destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ('mode','variant','wall_tokens_per_second','peak_allocated_bytes','peak_reserved_bytes')}),flush=True)


def moe_worker(args,cfg):
    from src.moe.router import FreeMoE
    manifest=json.loads(Path('results/prompt3/transition_checkpoint_manifest.json').read_text())
    entry=manifest['canonical_transition_checkpoints']['sparse75']['model_weights']
    if hash_file(entry['path'])!=entry['sha256']:raise ValueError('Canonical weights mismatch')
    state=torch.load(entry['path'],map_location='cpu',weights_only=True)
    layer=FreeMoE(cfg['model']).cuda()
    layer.load_state_dict({name[len('blocks.3.moe.'):]:value for name,value in state.items() if name.startswith('blocks.3.moe.')});del state
    rows=[]
    for distribution in ('balanced','skewed'):
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        tokens=args.batch*args.context;ids=torch.arange(tokens,device='cuda')%layer.n
        if distribution=='skewed':ids[:int(tokens*.9)]=0
        hidden=torch.randn(tokens,cfg['model']['d_model'],device='cuda')*.01
        hidden[torch.arange(tokens,device='cuda'),ids]+=2;hidden=hidden.reshape(args.batch,args.context,-1)
        with torch.no_grad():layer.router.weight.zero_();layer.router.weight[:,:layer.n].copy_(torch.eye(layer.n,device='cuda'))
        x=hidden.detach().clone().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16):y,aux=layer(x)
        (y.square().mean()+aux*.01).backward()
        parity=dict(output=y.detach().cpu(),input_gradient=x.grad.detach().cpu(),
                    gradients={n:p.grad.detach().cpu() if p.grad is not None else None for n,p in layer.named_parameters()},
                    routes=layer.last['experts'].cpu())
        base=Path(args.worker_output);base.parent.mkdir(parents=True,exist_ok=True)
        torch.save(parity,base.with_name(base.stem+'_'+distribution+'.pt'));del x,y,aux,parity
        def step():
            layer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):value,auxiliary=layer(hidden)
            (value.square().mean()+auxiliary*.01).backward()
        for _ in range(3):step()
        layer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(12):
            start=time.perf_counter();step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
        rows.append(dict(distribution=distribution,median_forward_backward_seconds=statistics.median(times),
                         tokens_per_second=tokens/statistics.median(times),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(),padded_capacity=layer.last['padded_capacity'],
                         load=layer.last['load'].cpu().tolist()))
    report=dict(mode=args.mode,results=rows,canonical_file_unchanged=hash_file(entry['path'])==entry['sha256'],
                gpu=torch.cuda.get_device_name(),context=args.context,microbatch=args.batch,bf16=True,
                scope='Isolated Top1 MoE, checkpoint expert weights, forced matched balanced/90%-skewed routing, no optimizer steps. Three warmup and twelve measured forward/backward repetitions. Padded GEMM geometry and the one counts host read are retained.')
    Path(args.worker_output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report),flush=True)


def difference(a,b):
    d=(a.float()-b.float()).abs()
    return dict(max_abs=float(d.max()),mean_abs=float(d.mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--source-root');p.add_argument('--worker-output')
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--mode',choices=('before','after'))
    p.add_argument('--moe-only',action='store_true')
    p.add_argument('--reference-ref',default=BASELINE);p.add_argument('--steps',type=int,default=6)
    p.add_argument('--context',type=int,default=1024);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--chunk',type=int,default=1024);p.add_argument('--native',action='store_true')
    p.add_argument('--output',default='results/implementation_fixes/rtx5070.json')
    args=p.parse_args()
    if not 1<=args.steps<=20:raise ValueError('Benchmark limited to 1..20 measured steps')
    if args.worker:worker(args);return
    if Path(args.output).exists():raise FileExistsError('Use a new report version')
    reference=Path('work/implementation_reference')/args.reference_ref
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',args.reference_ref,'src','verify_install.py'],text=True).splitlines()
    for name in paths:
        path=reference/name;path.parent.mkdir(parents=True,exist_ok=True)
        raw=subprocess.check_output(['git','show',args.reference_ref+':'+name])
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Reference snapshot changed')
        if not path.exists():path.write_bytes(raw)
    results=[]; parity={}
    if args.moe_only:
        row={}
        for mode in ('before','after'):
            output=Path('work/implementation_benchmark')/f'moe_{mode}.json'
            subprocess.run([sys.executable,'-m','tools.implementation_benchmark','--worker','--moe-only',
                       '--source-root',str(reference if mode=='before' else Path.cwd()),'--worker-output',str(output),
                       '--mode',mode,'--variant','sparse75','--batch',str(args.batch),'--context',str(args.context)],check=True)
            row[mode]=json.loads(output.read_text())
        for distribution in ('balanced','skewed'):
            a,b=[torch.load(Path('work/implementation_benchmark')/f'moe_{mode}_{distribution}.pt',weights_only=True) for mode in ('before','after')]
            differences=[difference(a[k],b[k]) for k in ('output','input_gradient')]
            for name,ga in a['gradients'].items():
                gb=b['gradients'][name]
                if ga is None and gb is None:continue
                if ga is None:ga=torch.zeros_like(gb)
                if gb is None:gb=torch.zeros_like(ga)
                differences.append(difference(ga,gb))
            parity[distribution]=dict(max_abs=max(d['max_abs'] for d in differences),routes_exact=torch.equal(a['routes'],b['routes']))
        passed=all(r['max_abs']<=.002 and r['routes_exact'] for r in parity.values())
        destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(dict(reference_commit=args.reference_ref,**row,parity=parity,passed=passed),indent=2)+'\n',encoding='utf8')
        print(json.dumps({'passed':passed,'output':args.output,'parity':parity}),flush=True)
        if not passed:raise RuntimeError('Isolated MoE parity failed')
        return
    for variant in VARIANTS:
        row={}
        for mode in ('before','after'):
            output=Path('work/implementation_benchmark')/f'{variant}_{mode}_b{args.batch}_c{args.chunk}.json'
            command=[sys.executable,'-m','tools.implementation_benchmark','--worker','--source-root',str(reference if mode=='before' else Path.cwd()),
                     '--worker-output',str(output),'--variant',variant,'--mode',mode,'--steps',str(args.steps),
                     '--context',str(args.context),'--batch',str(args.batch),'--chunk',str(args.chunk)]
            if args.native and mode=='after':command.append('--native')
            subprocess.run(command,check=True);row[mode]=json.loads(output.read_text())
        paths=[Path('work/implementation_benchmark')/f'{variant}_{mode}_b{args.batch}_c{args.chunk}.pt' for mode in ('before','after')]
        a,b=[torch.load(path,map_location='cpu',weights_only=True) for path in paths]
        gradients={}
        for name,ga in a['gradients'].items():
            gb=b['gradients'][name]
            if ga is None and gb is None:continue
            if ga is None:ga=torch.zeros_like(gb)
            if gb is None:gb=torch.zeros_like(ga)
            gradients[name]=difference(ga,gb)
        logits=difference(a['logits'],b['logits'])
        parity[variant]=dict(logits=logits,loss_abs=abs(a['loss']-b['loss']),
                    routes_exact=all(torch.equal(v,b['routes'][i]) for i,v in a['routes'].items()),
                    gradient_max_abs=max(d['max_abs'] for d in gradients.values()),gradient_max_mean=max(d['mean_abs'] for d in gradients.values()),
                    tolerances=dict(logits_max_abs=.125,logits_mean_abs=.005,gradient_max_abs=.003,loss_abs=.005))
        parity[variant]['passed']=(logits['max_abs']<=.125 and logits['mean_abs']<=.005
                    and parity[variant]['gradient_max_abs']<=.003 and parity[variant]['loss_abs']<=.005 and parity[variant]['routes_exact'])
        results.append(dict(variant=variant,**row));del a,b;gc.collect()
    report=dict(reference_commit=args.reference_ref,results=results,initial_forward_backward_parity=parity,
                passed=all(r['passed'] for r in parity.values()) and all(r[m]['canonical_file_unchanged'] for r in results for m in ('before','after')),
                source_sha256={path:hash_file(path) for path in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','src','tools/implementation_benchmark.py'],text=True).splitlines() if Path(path).is_file()})
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':args.output,'passed':report['passed'],'parity':parity}),flush=True)
    if not report['passed']:raise RuntimeError('Parity failed; promotion blocked')


if __name__=='__main__':main()

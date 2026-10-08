"""Bounded RTX 5070 speed measurements; never a quality-training launcher."""
from __future__ import annotations
import argparse, copy, csv, gc, json, statistics, time
from pathlib import Path
import torch
from src.config import load_config
from src.model import LanguageModel
from src.moe.router import FreeMoE
from src.training.engine import amp, make_optimizer, require_cuda, read_utilization, seed_all
from src.training.data import PackedStream
from tools.count_params import count_model

def gate():
    p=Path('results/moe_backend_parity.json')
    if not p.exists() or not json.loads(p.read_text()).get('passed'): raise RuntimeError('Parity required before benchmark promotion')
    from tools.moe_parity import source_hashes
    if json.loads(p.read_text()).get('source_sha256')!=source_hashes(): raise RuntimeError('Source changed since parity; rerun gate')

def measure(function,warm=3,repeats=10):
    for _ in range(warm): function()
    torch.cuda.synchronize(); times=[]; device=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter(); a.record(); function(); b.record(); b.synchronize()
        times.append((time.perf_counter()-start)*1000); device.append(a.elapsed_time(b))
    return statistics.median(times),statistics.median(device)

def write(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)

def micro():
    gate(); cfg=load_config('configs/runtime/sparse_reference.yaml'); seed_all(42)
    whole=LanguageModel(cfg).cuda().eval()
    state=torch.load('experiments/phase1a_sparse/checkpoints/last.pt',map_location='cpu',weights_only=False)
    whole.load_state_dict(state['model']); layer=whole.blocks[2].moe
    weights=copy.deepcopy(layer.state_dict()); rows=[]
    for context in (256,1024,2048):
        for batch in (1,2,4,8):
            stream=PackedStream(Path(cfg['data']['shards']),'val',context,42); x,_=stream.next(batch,'cuda')
            captured=[]; hook=layer.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
            with torch.no_grad(),amp(cfg): whole(x)
            hook.remove(); natural=captured[0]; n=batch*context
            for distribution in ('balanced','moderate','strong','natural'):
                for backend in ('reference','grouped'):
                    m=FreeMoE(dict(cfg['model'],moe_backend=backend)).cuda(); m.load_state_dict(weights)
                    if distribution=='natural': hidden=natural.clone()
                    else:
                        seed_all(42); ids=torch.arange(n,device='cuda')%m.n
                        fraction={'balanced':0.,'moderate':.30,'strong':.90}[distribution]
                        if fraction: ids[:int(n*fraction)]=0
                        ids=ids[torch.randperm(n,device='cuda')]
                        hidden=torch.randn(n,cfg['model']['d_model'],device='cuda')*.01
                        hidden[torch.arange(n,device='cuda'),ids]+=2; hidden=hidden.view(batch,context,-1)
                        with torch.no_grad(): m.router.weight.zero_(); m.router.weight[:,:m.n].copy_(torch.eye(m.n,device='cuda'))
                    def forward():
                        m.eval()
                        with torch.no_grad(),amp(cfg): m(hidden)
                    def backward():
                        m.train(); m.zero_grad(set_to_none=True)
                        with amp(cfg): y,aux=m(hidden); loss=y.square().mean()+.01*aux
                        loss.backward()
                    torch.cuda.reset_peak_memory_stats(); f,fe=measure(forward); fb,fbe=measure(backward)
                    counts=m.last['load'].tolist(); capacity=m.last['padded_capacity']
                    rows.append(dict(context=context,microbatch=batch,tokens=n,distribution=distribution,backend=backend,
                                     forward_ms=f,forward_cuda_elapsed_ms=fe,forward_backward_ms=fb,forward_backward_cuda_elapsed_ms=fbe,
                                     forward_tok_s=n*1000/f,forward_backward_tok_s=n*1000/fb,
                                     allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),
                                     expert_counts=json.dumps(counts),nonempty_experts=sum(c>0 for c in counts),
                                     padded_capacity=capacity,padding_ratio=m.n*capacity/n if capacity else None,
                                     gpu_utilization=None,gpu_utilization_scope='Short case; not a reliable occupancy measurement',
                                     repeats=10,forward_speedup_vs_reference=None,forward_backward_speedup_vs_reference=None))
                    del m; gc.collect()
                a,b=rows[-2:]; b['forward_speedup_vs_reference']=a['forward_ms']/b['forward_ms']; b['forward_backward_speedup_vs_reference']=a['forward_backward_ms']/b['forward_backward_ms']
            write('results/moe_microbench.csv',rows)
            print(json.dumps({'context':context,'batch':batch,'cases_completed':len(rows)}),flush=True)
    return rows

def full_one(config,tag,repetition,steps=20,fresh=False):
    seed_all(42); cfg=load_config(config); model=LanguageModel(cfg).cuda()
    source='memory' if cfg['memory']['enabled'] else 'dense' if not cfg['model']['moe_layers'] else 'sparse'
    # Different geometry is a fresh, short learning smoke, not a weight conversion.
    geometry_changed=cfg['model']['experts']!=12 or cfg['model']['expert_ffn']!=160
    if not fresh:
        state=torch.load(f'experiments/phase1a_{source}/checkpoints/last.pt',map_location='cpu',weights_only=False)
        if not geometry_changed or not cfg['model']['moe_layers']: model.load_state_dict(state['model'])
        del state
    opt,scheduler=make_optimizer(model,cfg)
    stream=PackedStream(Path(cfg['data']['shards']),'train',1024,42)
    fixed=[stream.next(2,'cuda') for _ in range(4)]
    def forward():
        model.eval()
        with torch.no_grad(),amp(cfg): model(fixed[0][0])
    def backward():
        model.train(); opt.zero_grad()
        with amp(cfg): loss=model(*fixed[0])['loss']
        loss.backward()
    f,fe=measure(forward,repeats=6); fb,fbe=measure(backward,repeats=6)
    def step(batches):
        model.train(); opt.zero_grad(set_to_none=True); total=0.
        for x,y in batches:
            with amp(cfg): out=model(x,y); loss=out['loss']/4
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite smoke loss')
            loss.backward(); total+=float(out['lm_loss'].detach())
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True))
        opt.step(); scheduler.step(); torch.cuda.synchronize()
        return total/4,norm
    warm_losses=[step(fixed)[0] for _ in range(3)]
    torch.cuda.reset_peak_memory_stats(); step_times=[]; losses=[]
    for _ in range(steps):
        start=time.perf_counter(); value,norm=step(fixed); step_times.append(time.perf_counter()-start); losses.append(value)
    pure_seconds=sum(step_times); tokens=steps*8192
    # A distinct matched loop includes dataset batching and lightweight JSONL logs.
    stream=PackedStream(Path(cfg['data']['shards']),'train',1024,42)
    path=Path(f'results/moe_{tag}_rep{repetition}_benchmark.jsonl')
    with path.open('w') as log:
        start=time.perf_counter()
        for i in range(steps):
            batches=[stream.next(2,'cuda') for _ in range(4)]; loss,norm=step(batches)
            log.write(json.dumps({'step':i+1,'tokens':(i+1)*8192,'loss':loss,'grad_norm':norm})+'\n'); log.flush()
        wall_seconds=time.perf_counter()-start
    count=count_model(model); row=dict(model=cfg['name'],tag=tag,repetition=repetition,context=1024,microbatch=2,accumulation=4,
        measured_steps_per_loop=steps,tokens_per_loop=tokens,training_step_seconds=pure_seconds,training_step_tok_s=tokens/pure_seconds,
        wall_seconds=wall_seconds,wall_tok_s=tokens/wall_seconds,forward_ms=f,forward_backward_ms=fb,
        forward_cuda_elapsed_ms=fe,forward_backward_cuda_elapsed_ms=fbe,
        allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),
        gpu_utilization_at_end=read_utilization(),stored_params=count['total'],routed_params=count['routed_experts'],active_params_est=count['active_estimate'],
        experts=cfg['model']['experts'],expert_ffn=cfg['model']['expert_ffn'],smoke_loss_first=warm_losses[0],smoke_loss_last=losses[-1],
        optimizer_state='fresh AdamW for speed/smoke only, not quality continuation',initial_weights='fresh geometry, seed 42' if fresh or geometry_changed and cfg['model']['moe_layers'] else 'existing Phase1A trained checkpoint',
        scope='Pure loop uses preloaded identical batches; wall loop follows separately and includes data/JSONL but excludes initialization, checkpoint/evaluation. Both loops use equal steps/policy across models. No quality-ranking claim.')
    del model,opt,scheduler,fixed; gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(row),flush=True); return row

def full():
    gate(); configs=[('configs/runtime/dense.yaml','dense'),('configs/runtime/sparse_reference.yaml','reference'),('configs/runtime/sparse_grouped.yaml','grouped'),('configs/runtime/sparse_grouped_memory.yaml','memory')]
    rows=[]
    for rep in (1,2):
        for config,tag in (configs if rep==1 else list(reversed(configs))):
            rows.append(full_one(config,tag,rep)); write('results/moe_fullmodel_benchmark.csv',rows)
    return rows

def granularity():
    gate(); rows=[]
    configs=[('configs/runtime/sparse_grouped.yaml','geometry12'),('configs/runtime/experts6.yaml','geometry6'),('configs/runtime/experts4.yaml','geometry4')]
    for rep in (1,2):
        for config,tag in (configs if rep==1 else list(reversed(configs))):
            row=full_one(config,tag,rep,steps=10,fresh=True)
            row['comparison_basis']='Routed stored FFN capacity held at 5,529,600 parameters: E*FFN=1920 in 3 layers; router size and active per-token FFN differ. All fresh seed42, same data/optimizer; timing geometry, not quality ranking.'
            rows.append(row); write('results/moe_expert_granularity.csv',rows)
    return rows

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['micro','full','granularity']); args=parser.parse_args()
    require_cuda('cuda'); torch.backends.cuda.matmul.allow_tf32=True
    globals()[args.mode]()

if __name__=='__main__': main()

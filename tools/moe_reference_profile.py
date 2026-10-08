"""Bounded, annotated copy of the unchanged reference computation for attribution."""
from __future__ import annotations
import json, sys, time, types
from pathlib import Path
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.config import load_config
from src.model import LanguageModel
from src.training.data import PackedStream
from src.training.engine import amp, require_cuda, make_optimizer, read_utilization

def instrumented(self, x):
    shape=x.shape; flat=x.reshape(-1,shape[-1])
    with record_function('moe/router'): routed=self.router(flat)
    with record_function('moe/softmax_topk'):
        probs=routed.float().softmax(-1); weights,indices=probs.topk(self.k,dim=-1)
        if self.k>1: weights=weights/weights.sum(-1,keepdim=True)
    if self.backend=='grouped':
        from src.moe.grouped import dispatch
        with record_function('moe/grouping_indexing'):
            integers=torch.zeros(self.n,dtype=torch.long,device=flat.device).scatter_add_(0,indices[:,0],torch.ones(flat.shape[0],dtype=torch.long,device=flat.device))
        with record_function('moe/grouped_dispatch'): output,capacity=dispatch(flat,indices,weights,self.experts,integers,self._expert_up,self._expert_down)
        self.active_since_zero.logical_or_(integers>0)
    else:
        output=torch.zeros_like(flat)
        with record_function('moe/expert_iteration'):
            for expert_id,expert in enumerate(self.experts):
                with record_function('moe/grouping_indexing'):
                    token,slot=torch.where(indices==expert_id)
                    selected=flat[token] if token.numel() else None
                if token.numel():
                    with record_function('moe/expert_gemms_swiglu'): values=expert(selected)
                    with record_function('moe/scatter_combine'):
                        values=values*weights[token,slot,None].to(flat.dtype)
                        output.index_add_(0,token,values.to(output.dtype))
    with record_function('moe/balance_diagnostics'):
        counts=integers.float() if self.backend=='grouped' else torch.bincount(indices.flatten(),minlength=self.n).float(); load=counts/indices.numel()
        aux=self.n*(load.detach()*probs.mean(0)).sum()
        self.last={'experts':indices.detach().view(*shape[:-1],self.k),'probabilities':probs.detach().view(*shape[:-1],self.n),
                   'selected_weights':weights.detach().view(*shape[:-1],self.k),
                   'entropy':-(probs.detach()*probs.detach().clamp_min(1e-9).log()).sum(-1).mean(),
                   'load':counts.detach(),'imbalance':counts.std(unbiased=False)/counts.mean().clamp_min(1)}
    return output.view(shape),aux

def main():
    require_cuda('cuda'); torch.manual_seed(42)
    grouped='--grouped' in sys.argv
    cfg=load_config('configs/runtime/sparse_grouped.yaml' if grouped else 'configs/phase1a/sparse.yaml'); model=LanguageModel(cfg).cuda()
    state=torch.load('experiments/phase1a_sparse/checkpoints/last.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']); opt,_=make_optimizer(model,cfg)
    stream=PackedStream(Path(cfg['data']['shards']),'train',1024,42)
    x,y=stream.next(2,'cuda'); original=[]
    for block in model.blocks:
        if block.moe is not None:
            original.append((block.moe,block.moe.forward)); block.moe.forward=types.MethodType(instrumented,block.moe)
    def step():
        with record_function('training/zero_grad'): opt.zero_grad(set_to_none=True)
        for _ in range(4):
            with record_function('training/forward'),amp(cfg): loss=model(x,y)['loss']/4
            with record_function('training/backward'): loss.backward()
        with record_function('training/clip_optimizer'):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        with record_function('training/synchronize'): torch.cuda.synchronize()
    for _ in range(3): step()
    utilization_before=read_utilization(); start=time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as p:
        for _ in range(3): step()
    wall=time.perf_counter()-start
    averages=p.key_averages(); events=p.events()
    cpu=[e for e in averages if e.device_type==torch.autograd.DeviceType.CPU]
    device=[e for e in averages if e.device_type==torch.autograd.DeviceType.CUDA and not e.is_user_annotation]
    top=sorted(device,key=lambda e:e.self_device_time_total,reverse=True)[:25]
    components={e.key:{'cpu_total_ms':e.cpu_time_total/1000,'cpu_self_ms':e.self_cpu_time_total/1000,
                            'device_total_ms':e.device_time_total/1000,'calls':e.count} for e in cpu if e.key.startswith(('moe/','training/'))}
    device_total=sum(e.self_device_time_total for e in device)
    for value in components.values(): value['fraction_of_total_device_work']=value['device_total_ms']/(device_total/1000) if device_total else None
    report={'profiled_steps':3,'context':1024,'microbatch':2,'accumulation':4,'wall_seconds_with_profiler':wall,
            'components':components,'total_device_self_ms':device_total/1000,
            'top_device_events':[{'name':e.key,'device_self_ms':e.self_device_time_total/1000,'calls':e.count} for e in top],
            'cuda_events_observed':sum(e.device_type==torch.autograd.DeviceType.CUDA for e in events),
            'gpu_utilization_before':utilization_before,'gpu_utilization_after':read_utilization(),
            'expert_groups':{str(i):r['load'].tolist() for i,r in model.routes().items()},
            'backend':'grouped' if grouped else 'reference',
            'scope':'Annotated computation, trained Phase1A weights, three warmed optimizer steps. CPU/device scopes are nested and must not be summed; profiler overhead is not throughput. CUDA event count is not guaranteed to equal kernel launches. Python iteration CPU self excludes child operator work; synchronization includes operator-internal synchronization shown in top CPU events. Autograd backward device work is not reliably attributed to its enclosing CPU annotation: its device_total is NOT total backward CUDA work.'}
    report['top_cpu_events']=[{'name':e.key,'cpu_self_ms':e.self_cpu_time_total/1000,'calls':e.count} for e in sorted(cpu,key=lambda e:e.self_cpu_time_total,reverse=True)[:25]]
    report['profile_annotation_filter']='CPU scopes and CUDA non-annotation kernel events separated to avoid merging identically named CPU/CUDA annotations or double-counting operator/kernel totals.'
    Path('results/moe_grouped_profile.json' if grouped else 'results/moe_reference_profile.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()

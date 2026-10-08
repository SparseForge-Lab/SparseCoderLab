"""Bounded native-GQA and component/memory/checkpoint diagnostics."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch

from src.config import load_config
from src.model import LanguageModel
from src.model.layers import Attention
from src.training.engine import make_optimizer
from src.training.checkpoint import save_training, resume_training
from src.utils.hashing import sha256_file


def attention():
    cfg=load_config('configs/prompt3/sparse75.yaml')['model']
    torch.manual_seed(42);reference=Attention(cfg).cuda();native=Attention(dict(cfg,gqa_backend='native')).cuda()
    native.load_state_dict(reference.state_dict())
    x=torch.randn(8,1024,512,device='cuda')
    def run(module,backward=True):
        module.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): y=module(x)
        if backward:y.square().mean().backward()
        return y
    a=run(reference);ga=[p.grad.detach().clone() for p in reference.parameters()]
    b=run(native);gb=[p.grad.detach().clone() for p in native.parameters()]
    parity=dict(output_max_abs=float((a-b).abs().max().detach()),gradient_max_abs=max(float((u-v).abs().max()) for u,v in zip(ga,gb)),
                passed=bool(torch.allclose(a,b,atol=.002,rtol=.03)) and all(torch.allclose(u,v,atol=.002,rtol=.05) for u,v in zip(ga,gb)),
                atol=.002,output_rtol=.03,gradient_rtol=.05)
    del a,b,ga,gb
    results={}
    for name,module,other in (('repeat',reference,native),('native',native,reference)):
        other.zero_grad(set_to_none=True)
        for _ in range(3):run(module)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(12):
            started=time.perf_counter();run(module);torch.cuda.synchronize();times.append(time.perf_counter()-started)
        results[name]=dict(median_forward_backward_seconds=statistics.median(times),tokens_per_second=x.shape[0]*x.shape[1]/statistics.median(times),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return dict(parity=parity,results=results,context=1024,microbatch=8,
                scope='Isolated attention forward/backward, same weights/input, three warmups and twelve measured repetitions. Both attention weight sets and the fixed input remain resident. Native GQA is opt-in; whole-model promotion requires additional routing evidence.')


def tensor_hashes(model,opt,scheduler):
    result={}
    for name,parameter in model.named_parameters():
        for key,value in [('model',parameter)]+list(opt.state.get(parameter,{}).items()):
            if torch.is_tensor(value):
                raw=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                result[name+'/'+key]=hashlib.sha256(raw).hexdigest()
    result['scheduler']=scheduler.state_dict()
    return result


def components():
    cfg=load_config('configs/prompt3/sparse75_ngram25m.yaml');cfg['training']['loss_chunk_tokens']=1024
    manifest=json.loads(Path('results/prompt3/transition_checkpoint_manifest.json').read_text())
    entry=manifest['canonical_transition_checkpoints']['sparse75_ngram25m']['model_weights']
    if sha256_file(entry['path'])!=entry['sha256']:raise ValueError('Canonical identity mismatch')
    model=LanguageModel(cfg).cuda();model.load_state_dict(torch.load(entry['path'],map_location='cpu',weights_only=True))
    opt,sched=make_optimizer(model,cfg);x=torch.randint(0,32768,(8,1024),device='cuda');y=torch.roll(x,-1,1)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,y,return_outputs=False)
        loss=out['loss'];loss.backward();del out,loss
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();sched.step()
    for _ in range(2):step()
    records=[];original=[]
    def wrap(obj,method,label):
        function=getattr(obj,method);original.append((obj,method,function))
        def measured(*args,**kwargs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter();start.record();result=function(*args,**kwargs);end.record()
            records.append((label,start,end,time.perf_counter()-wall));return result
        setattr(obj,method,measured)
    for block in model.blocks:
        wrap(block.attention,'forward','attention')
        if block.moe is not None:wrap(block.moe.router,'forward','router_linear')
    wrap(model.memory,'forward','ngram')
    import src.moe.grouped as grouped
    wrap(grouped,'dispatch','moe_dispatch');wrap(opt,'step','optimizer')
    try:
        started=time.perf_counter();step();torch.cuda.synchronize();elapsed=time.perf_counter()-started
    finally:
        for obj,method,function in reversed(original):setattr(obj,method,function)
    summary={}
    for label,start,end,wall in records:
        row=summary.setdefault(label,dict(calls=0,cuda_interval_ms=0.,host_call_ms=0.))
        row['calls']+=1;row['cuda_interval_ms']+=start.elapsed_time(end);row['host_call_ms']+=wall*1000
    opt.zero_grad(set_to_none=True)
    state=tensor_hashes(model,opt,sched)
    checkpoint=Path('work/implementation_diagnostics/checkpoint.pt');checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if checkpoint.exists():raise FileExistsError('Use a clean diagnostics work path')
    started=time.perf_counter();save_training(checkpoint,model,opt,sched,dict(step=3,tokens_seen=3*x.numel(),cursor=3));save_seconds=time.perf_counter()-started
    size=checkpoint.stat().st_size;digest=sha256_file(checkpoint)
    started=time.perf_counter();meta=resume_training(checkpoint,model,opt,sched,'cuda');load_seconds=time.perf_counter()-started
    equal=state==tensor_hashes(model,opt,sched)
    checkpoint.unlink()  # This tool's own temporary smoke checkpoint only.
    return dict(step_seconds_with_instrumentation=elapsed,components=summary,checkpoint_save_seconds=save_seconds,
                checkpoint_resume_seconds=load_seconds,temporary_checkpoint_bytes=size,temporary_checkpoint_sha256=digest,
                model_optimizer_scheduler_hashes_equal_after_resume=equal,metadata_counters=(meta['step'],meta['tokens_seen'],meta['cursor']),
                canonical_file_unchanged=sha256_file(entry['path'])==entry['sha256'],
                scope='Single instrumented step after two warmups, canonical 25M-Ngram weights loaded read-only. CUDA intervals include stream gaps and host submission delays; host calls include any internal synchronization. Component intervals cover forward and optimizer, not attributed backward kernels. Fixed preloaded synthetic input means dataloader wait is excluded. Full temporary model/Adam/scheduler checkpoint is staged on CPU during resume and removed after hash verification.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/implementation_fixes/diagnostics_v1.json');args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError('Choose a new diagnostics version')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    result=dict(gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,native_gqa=attention())
    gc.collect();torch.cuda.empty_cache();result['training_components']=components()
    result['passed']=result['native_gqa']['parity']['passed'] and result['training_components']['model_optimizer_scheduler_hashes_equal_after_resume'] and result['training_components']['canonical_file_unchanged']
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('Diagnostics validation failed')


if __name__=='__main__':main()

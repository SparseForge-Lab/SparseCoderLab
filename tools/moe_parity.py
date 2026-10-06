from __future__ import annotations
import copy, hashlib, json, os
from pathlib import Path
import torch
from src.config import load_config
from src.model import LanguageModel
from src.moe.router import FreeMoE
from src.training.engine import make_optimizer, require_cuda

def source_hashes():
    paths=['src/moe/grouped.py','src/moe/router.py','src/training/moe_optimizer.py',
           'src/training/engine.py','src/model/layers.py','src/model/transformer.py','src/model/__init__.py']
    return {path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}

def diff(a,b):
    d=(a.detach().float()-b.detach().float()).abs()
    return {'max_abs':float(d.max()),'mean_abs':float(d.mean()),
            'relative_l2':float(d.norm()/a.detach().float().norm().clamp_min(1e-12))}

def trial(device,bf16,trained=False,control=False):
    torch.manual_seed(42); cfg=load_config('configs/test.yaml')
    if trained:
        cfg=load_config('configs/phase1a/sparse.yaml')
    elif bf16:
        cfg['model'].update(d_model=320,q_heads=5,kv_heads=1,head_dim=64,ffn=384,expert_ffn=160,experts=12)
    cfg['training']['fused_optimizer']=False
    ref=LanguageModel(cfg).to(device); gc=copy.deepcopy(cfg); gc['model']['moe_backend']='reference' if control else 'grouped'
    grouped=LanguageModel(gc).to(device); grouped.load_state_dict(ref.state_dict())
    opt,_=make_optimizer(ref,cfg); gopt,_=make_optimizer(grouped,gc)
    if trained:
        state=torch.load('experiments/phase1a_sparse/checkpoints/last.pt',map_location=device,weights_only=False)
        ref.load_state_dict(state['model']); grouped.load_state_dict(state['model'])
        opt.load_state_dict(copy.deepcopy(state['optimizer'])); gopt.load_state_dict(copy.deepcopy(state['optimizer'])); del state
        from src.training.data import PackedStream
        stream=PackedStream(Path(cfg['data']['shards']),'train',1024,42)
    reports=[]
    for step in range(5):
        if trained: x,target=stream.next(2,device)
        else: x=torch.randint(0,cfg['model']['vocab_size'],(2,64),device=device); target=torch.roll(x,-1,1)
        opt.zero_grad(); gopt.zero_grad()
        with torch.autocast(device,dtype=torch.bfloat16,enabled=bf16):
            a=ref(x,target); b=grouped(x,target)
        values={key:diff(a[key],b[key]) for key in ('logits','aux_loss','loss')}
        routing={str(i):{'expert_id_mismatches':int((ref.routes()[i]['experts']!=grouped.routes()[i]['experts']).sum()),
                         'probabilities':diff(ref.routes()[i]['probabilities'],grouped.routes()[i]['probabilities']),
                         'selected_weights':diff(ref.routes()[i]['selected_weights'],grouped.routes()[i]['selected_weights'])} for i in ref.routes()}
        a['loss'].backward(); b['loss'].backward()
        grads={}
        for (name,pa),(_,pb) in zip(ref.named_parameters(),grouped.named_parameters()):
            ga=pa.grad if pa.grad is not None else torch.zeros_like(pa)
            gb=pb.grad if pb.grad is not None else torch.zeros_like(pb)
            grads[name]=diff(ga,gb)
        opt.step(); gopt.step()
        updates={name:diff(pa,pb) for (name,pa),(_,pb) in zip(ref.named_parameters(),grouped.named_parameters())}
        reports.append({'step':step+1,'forward':values,'routing':routing,'gradients':grads,'parameters_after_step':updates})
    # Isolated MoE with identical input: routing must be exactly equal, even BF16.
    m=cfg['model']; ma=FreeMoE(m).to(device); mb=FreeMoE(dict(m,moe_backend='grouped')).to(device); mb.load_state_dict(ma.state_dict())
    hidden=torch.randn(2,64,m['d_model'],device=device)
    with torch.autocast(device,dtype=torch.bfloat16,enabled=bf16): ya,aa=ma(hidden); yb,ab=mb(hidden)
    isolated={'output':diff(ya,yb),'aux':diff(aa,ab),'ids_equal':torch.equal(ma.last['experts'],mb.last['experts']),
              'probs_equal':torch.equal(ma.last['probabilities'],mb.last['probabilities']),
              'weights_equal':torch.equal(ma.last['selected_weights'],mb.last['selected_weights'])}
    # Predeclared absolute tolerances reflect BF16's ~0.78% mantissa unit;
    # this is small, finite deterministic numerical parity, not bitwise claims.
    tolerances={'logits_max_abs':.05 if bf16 else 2e-5,'logits_mean_abs':.003 if bf16 else 2e-6,
                'gradient_max_abs':.003 if bf16 else 2e-5,'parameter_max_abs_after_5_steps':.0005 if bf16 else 2e-5}
    if trained:
        tolerances['logits_max_abs']=.125 # Two BF16 ULP at trained logits in [8,16].
        tolerances['logits_mean_abs']=.005
    passed=isolated['ids_equal'] and isolated['probs_equal'] and isolated['weights_equal']
    for r in reports:
        passed &= r['forward']['logits']['max_abs']<=tolerances['logits_max_abs'] and r['forward']['logits']['mean_abs']<=tolerances['logits_mean_abs']
        passed &= all(g['max_abs']<=tolerances['gradient_max_abs'] for g in r['gradients'].values())
        passed &= all(g['max_abs']<=tolerances['parameter_max_abs_after_5_steps'] for g in r['parameters_after_step'].values())
    return {'device':device,'bf16':bf16,'passed':bool(passed),'tolerances':tolerances,'isolated_moe':isolated,'optimizer_steps':reports,
            'scope':'Identical state and inputs. Shared/resident/router/expert gradients reported per tensor; BF16 rounding can propagate across layers/updates, IDs on an identical isolated input are exact. All expert parameter/state_dict names remain unchanged.'}

def main():
    import sys
    deterministic='--deterministic' in sys.argv
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
    require_cuda('cuda'); torch.backends.cuda.matmul.allow_tf32=False
    if '--control' in sys.argv:
        result=trial('cuda',True,trained=True,control=True)
        result['deterministic_algorithms']=deterministic
        suffix='_deterministic' if deterministic else ''
        Path(f'results/moe_reference_repeatability{suffix}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'passed':result['passed'],'control':'reference vs reference','steps':[{'step':s['step'],'logits':s['forward']['logits'],'embedding':s['parameters_after_step']['embedding.weight']} for s in result['optimizer_steps']]},indent=2))
        return
    report={'fp32_cpu':trial('cpu',False),'bf16_rtx5070':trial('cuda',True),'trained_checkpoint_rtx5070':trial('cuda',True,trained=True)}
    report['passed']=all(v['passed'] for v in report.values())
    report['deterministic_algorithms']=deterministic
    report['source_sha256']=source_hashes()
    report['scope']='Deterministic parity uses process-local CUBLAS_WORKSPACE_CONFIG and deterministic algorithms; native BF16 timing uses the original non-deterministic training policy. Historical failed controls are preserved separately.'
    Path('results/moe_backend_parity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:({'passed':v['passed'],'isolated_moe':v['isolated_moe']} if isinstance(v,dict) and 'passed' in v else v) for k,v in report.items()},indent=2))
    if not report['passed']: raise RuntimeError('Backend parity failed; performance promotion blocked')

if __name__=='__main__': main()

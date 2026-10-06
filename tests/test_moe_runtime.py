import copy
import pytest
import torch
from src.config import load_config
from src.moe.router import FreeMoE
from src.model import LanguageModel
from src.training.engine import make_optimizer
from src.training.moe_optimizer import GroupedAdamW

def pair(device='cpu'):
    cfg=load_config('configs/test.yaml'); m=cfg['model']; m.update(experts=4,expert_ffn=16,top_k=1)
    torch.manual_seed(12); ref=FreeMoE(m).to(device)
    grouped=FreeMoE(dict(m,moe_backend='grouped')).to(device); grouped.load_state_dict(ref.state_dict())
    return ref,grouped

def input_for(distribution,device='cpu'):
    ids={'balanced':[0,1,2,3]*5,'empty':[0,1]*10,'collapse':[0]*19+[1]}[distribution]
    torch.manual_seed(4); x=torch.randn(1,len(ids),32,device=device)*.01
    x[0,torch.arange(len(ids),device=device),torch.tensor(ids,device=device)]+=2
    return x

def force_router(model):
    with torch.no_grad():
        model.router.weight.zero_(); model.router.weight[:,:4].copy_(torch.eye(4,device=model.router.weight.device))

@pytest.mark.parametrize('distribution',['balanced','empty','collapse'])
def test_grouped_top1_forward_grad_and_routing(distribution):
    a,b=pair(); force_router(a); force_router(b); x=input_for(distribution)
    xa=x.clone().requires_grad_(); xb=x.clone().requires_grad_()
    ya,la=a(xa); yb,lb=b(xb)
    torch.testing.assert_close(ya,yb,atol=1e-6,rtol=1e-5)
    for key in ('experts','probabilities','selected_weights','load','entropy','imbalance'):
        assert torch.equal(a.last[key],b.last[key])
    assert torch.equal(la,lb)
    (ya.square().mean()+.01*la).backward(); (yb.square().mean()+.01*lb).backward()
    torch.testing.assert_close(xa.grad,xb.grad,atol=1e-6,rtol=1e-5)
    for pa,pb in zip(a.parameters(),b.parameters()):
        ga=pa.grad if pa.grad is not None else torch.zeros_like(pa)
        torch.testing.assert_close(ga,pb.grad,atol=1e-6,rtol=1e-5)

def test_grouped_optimizer_preserves_empty_expert_momentum_and_decay():
    a,b=pair(); force_router(a); force_router(b)
    oa=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.1)
    ob=GroupedAdamW(b,lr=.001,weight_decay=.1)
    for distribution in ['balanced','empty','collapse','balanced','empty','collapse']:
        x=input_for(distribution)
        for m,o in ((a,oa),(b,ob)):
            o.zero_grad(); y,aux=m(x); (y.square().mean()+.01*aux).backward(); o.step()
        for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa,pb,atol=1e-6,rtol=1e-5)
    # Previously active but absent experts must retain per-parameter Adam steps.
    for pa,pb in zip(a.parameters(),b.parameters()):
        assert torch.equal(oa.state[pa]['step'],ob.state[pb]['step'])

def test_accumulation_uses_union_of_experts():
    a,b=pair(); force_router(a); force_router(b)
    oa=torch.optim.AdamW(a.parameters(),lr=.001); ob=GroupedAdamW(b,lr=.001)
    for m,o in ((a,oa),(b,ob)):
        o.zero_grad()
        for distribution in ('balanced','empty'):
            y,aux=m(input_for(distribution)); (y.square().mean()+.01*aux).backward()
        o.step()
    for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa,pb,atol=1e-6,rtol=1e-5)

def test_reference_grouped_checkpoint_roundtrip(tmp_path):
    cfg=load_config('configs/test.yaml'); a=LanguageModel(cfg)
    other=copy.deepcopy(cfg); other['model']['moe_backend']='grouped'; b=LanguageModel(other)
    b.load_state_dict(a.state_dict()); path=tmp_path/'grouped.pt'; torch.save(b.state_dict(),path)
    restored=LanguageModel(cfg); restored.load_state_dict(torch.load(path,weights_only=True))
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in a.state_dict().items())
    assert list(a.state_dict())==list(b.state_dict())
    assert [n for n,_ in a.named_parameters()]==[n for n,_ in b.named_parameters()]

@pytest.mark.gpu
def test_grouped_bf16_gpu_forward_backward():
    a,b=pair('cuda'); force_router(a); force_router(b); x=input_for('balanced','cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        ya,la=a(x); yb,lb=b(x)
    assert torch.equal(a.last['experts'],b.last['experts'])
    torch.testing.assert_close(ya,yb,atol=.002,rtol=.03)
    (ya.square().mean()+.01*la).backward(); (yb.square().mean()+.01*lb).backward()
    for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa.grad,pb.grad,atol=.002,rtol=.05)

def test_top2_remains_reference_only():
    cfg=load_config('configs/test.yaml')['model']; cfg.update(top_k=2,moe_backend='grouped')
    with pytest.raises(ValueError,match='Top1'): FreeMoE(cfg)

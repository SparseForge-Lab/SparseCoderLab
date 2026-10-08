import pytest
import torch
from src.config import load_config
from src.model import LanguageModel
from tools.generate_canonical_python import last_logits


@pytest.mark.parametrize('device,bf16',[('cpu',False),pytest.param('cuda',False,marks=pytest.mark.gpu),
    pytest.param('cuda',True,marks=pytest.mark.gpu)])
def test_right_padding_and_other_rows_do_not_change_real_prefix_logits(device,bf16):
    cfg=load_config('configs/test.yaml');cfg['mtp']['enabled']=False
    cfg['memory']['enabled']=True;cfg['model']['moe_backend']='grouped'
    torch.manual_seed(123);model=LanguageModel(cfg).to(device).eval()
    sequences=[torch.randint(1,512,(n,),device=device) for n in (7,11,19)]
    padded=torch.zeros((3,19),dtype=torch.long,device=device)
    for i,ids in enumerate(sequences):padded[i,:len(ids)]=ids
    with torch.no_grad(),torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=bf16):
        result=model(padded)['logits']
        selected=last_logits(model,padded,torch.tensor([len(ids)-1 for ids in sequences],device=device))
        for i,ids in enumerate(sequences):
            reference=model(ids[None])['logits'][0]
            torch.testing.assert_close(result[i,:len(ids)],reference,atol=.003 if bf16 else 2e-6,rtol=.03 if bf16 else 2e-5)
            torch.testing.assert_close(selected[i],reference[-1],atol=.003 if bf16 else 2e-6,rtol=.03 if bf16 else 2e-5)

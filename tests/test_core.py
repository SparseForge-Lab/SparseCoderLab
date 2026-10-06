from __future__ import annotations
import copy, json, random
from pathlib import Path
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from src.config import load_config
from src.model import LanguageModel
from src.memory.ngram import stable_hash, NgramMemory
from src.moe.router import FreeMoE
from src.mtp.draft import horizon_targets
from src.training.checkpoint import save_training, resume_training
from src.training.data import LANGUAGE_SAMPLES, split_document, PackedStream
from src.training.engine import require_cuda, make_optimizer
from tools.count_params import count_model

@pytest.fixture
def cfg(): return load_config('configs/test.yaml')

def test_tokenizer_roundtrip():
    tok = Tokenizer.from_file('data/tokenizer.json')
    assert tok.get_vocab_size() == 32768
    for text in list(LANGUAGE_SAMPLES.values()) + ['\t\n    a  b\r\n', '😺 漢字 é\x00', '<compact> <mem:000123>']:
        assert tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text

def test_split_and_no_leakage():
    assert split_document('identical content', 42, .05) == split_document('identical content', 42, .05)
    train = {json.loads(x)['sha256'] for x in Path('data/shards/train_documents.jsonl').read_text().splitlines()}
    val = {json.loads(x)['sha256'] for x in Path('data/shards/val_documents.jsonl').read_text().splitlines()}
    assert train and val and train.isdisjoint(val)

def test_hash_independent_reference():
    tokens = torch.tensor([[1, 2, 9, 17]])
    got = stable_hash(tokens, 3, 256, 17)
    expected = []
    for t in range(4):
        h = 17
        for pos in range(t - 2, t + 1): h = (h * 65599 + (int(tokens[0, pos]) + 1 if pos >= 0 else 0)) % 2147483647
        expected.append(h % 256)
    assert got.tolist() == [expected]
    assert torch.equal(got, stable_hash(tokens.clone(), 3, 256, 17))

def test_memory_gradient_and_causality(cfg):
    memory = NgramMemory(cfg['memory'], 32); x = torch.randn(2, 8, 32, requires_grad=True); tokens = torch.randint(0, 512, (2, 8))
    memory(x, tokens).square().mean().backward()
    assert all(t.weight.grad is not None and t.weight.grad.abs().sum() > 0 for t in memory.tables)
    assert memory.projection.weight.grad.abs().sum() > 0
    changed = tokens.clone(); changed[:, 4:] = 0
    assert torch.equal(memory.bucket_ids(tokens)[0][:, :4], memory.bucket_ids(changed)[0][:, :4])

@pytest.mark.parametrize('k', [1, 2])
def test_moe_topk_load_gradients(cfg, k):
    cfg['model']['top_k'] = k; moe = FreeMoE(cfg['model']); x = torch.randn(2, 9, 32, requires_grad=True)
    y, aux = moe(x); (y.square().mean() + .01 * aux).backward()
    assert y.shape == x.shape and moe.last['experts'].shape == (2, 9, k)
    assert int(moe.last['load'].sum()) == 18 * k
    assert torch.isfinite(aux) and moe.router.weight.grad.abs().sum() > 0
    if k == 2: assert torch.allclose(moe.last['selected_weights'].sum(-1), torch.ones(2, 9))

def test_balancing_penalizes_collapse(cfg):
    moe = FreeMoE(cfg['model']); x = torch.ones(1, 8, 32)
    with torch.no_grad(): moe.router.weight.zero_(); moe.router.weight[0].fill_(1)
    _, collapsed = moe(x)
    with torch.no_grad(): moe.router.weight.zero_()
    _, uniform_probability = moe(x)
    assert collapsed > uniform_probability + 1

@pytest.mark.parametrize('sparse', [False, True])
def test_model_forward_backward_and_causal(cfg, sparse):
    if not sparse: cfg['model']['moe_layers'] = []
    model = LanguageModel(cfg); tokens = torch.randint(0, 512, (2, 12)); target = torch.roll(tokens, -1, 1)
    result = model(tokens, target); result['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    changed = tokens.clone(); changed[:, 7:] = 0
    assert torch.allclose(model(tokens)['logits'][:, :7], model(changed)['logits'][:, :7], atol=1e-6)

def test_mtp_shift_teacher_forcing_generation(cfg):
    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    assert horizon_targets(tokens, 1).tolist() == [[1, 2, 3, 4]]
    assert horizon_targets(tokens, 3).tolist() == [[3, 4]]
    cfg['mtp']['enabled'] = True; model = LanguageModel(cfg); x = torch.randint(0, 512, (2, 8))
    out = model(x, torch.roll(x, -1, 1), mtp_tokens=x); out['loss'].backward()
    assert set(out['mtp_metrics']) == {f't+{h}_{m}' for h in (1, 2, 3) for m in ('loss', 'accuracy')}
    generated = model.mtp.generate(out['hidden'][:, -1], x[:, -1], model.embedding)
    assert generated.shape == (2, 3) and generated.min() >= 0 and generated.max() < 512
    for design in ['shared_mtp_3step', 'three_layer_mtp']:
        cfg['mtp']['design'] = design; other = LanguageModel(cfg)
        assert len(other.mtp.blocks) == (1 if design.startswith('shared') else 3)

def test_checkpoint_resume_equivalence(cfg, tmp_path):
    torch.manual_seed(17); model = LanguageModel(cfg); optimizer, scheduler = make_optimizer(model, cfg)
    def step(m, o, s):
        x = torch.randint(0, 512, (2, 8)); o.zero_grad(); m(x, torch.roll(x, -1, 1))['loss'].backward(); o.step(); s.step()
        return x
    step(model, optimizer, scheduler)
    path = tmp_path / 'last.pt'; save_training(path, model, optimizer, scheduler, {'cursor': 7, 'tokens_seen': 112})
    expected_x = step(model, optimizer, scheduler); expected = copy.deepcopy(model.state_dict())
    restored = LanguageModel(cfg); o2, s2 = make_optimizer(restored, cfg)
    meta = resume_training(path, restored, o2, s2, 'cpu'); actual_x = step(restored, o2, s2)
    assert meta['cursor'] == 7 and meta['tokens_seen'] == 112 and torch.equal(expected_x, actual_x)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in expected.items())
    assert scheduler.state_dict() == s2.state_dict()

def test_dataloader_cursor_resume():
    a = PackedStream(Path('data/shards'), 'train', 32, 42); a.next(3, 'cpu'); cursor = a.cursor
    expected = a.next(4, 'cpu'); b = PackedStream(Path('data/shards'), 'train', 32, 42, cursor)
    actual = b.next(4, 'cpu'); assert all(torch.equal(x, y) for x, y in zip(expected, actual))

def test_shard_boundary_slice(tmp_path):
    from src.training.data import ShardedTokens
    for i in range(3): np.arange(i*5,(i+1)*5,dtype=np.uint16).tofile(tmp_path/f'{i}.bin')
    tokens=ShardedTokens(sorted(tmp_path.glob('*.bin')))
    assert len(tokens)==15 and tokens[3:12].tolist()==list(range(3,12))

def test_exact_counts():
    for path in ['dense_compute', 'dense_size', 'sparse_v3', 'sparse_memory']:
        c = load_config(f'configs/{path}.yaml'); model = LanguageModel(c); count = count_model(model); m = c['model']
        expected = m['vocab_size'] * m['d_model'] + m['layers'] * (m['d_model'] * (m['q_heads'] + 2 * m['kv_heads']) * m['head_dim'] + m['d_model']**2 + 3 * m['d_model'] * m['ffn'] + 2 * m['d_model']) + m['d_model']
        expected += len(m['moe_layers']) * (3 * m['d_model'] * m['expert_ffn'] * m['experts'] + m['d_model'] * m['experts'])
        if c['memory']['enabled']: expected += 4 * 131072 * 8 + 32 * 320 + 321
        assert count['total'] == expected and sum(p.numel() for p in model.parameters()) == expected

def test_no_silent_cpu():
    with pytest.raises(RuntimeError, match='CUDA'): require_cuda('cpu')

@pytest.mark.gpu
def test_gpu_bf16_and_hash_parity(cfg):
    require_cuda('cuda'); model = LanguageModel(cfg).cuda(); tokens = torch.randint(0, 512, (2, 12), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16): out = model(tokens, torch.roll(tokens, -1, 1))
    out['loss'].backward(); assert torch.isfinite(out['loss']) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert torch.equal(stable_hash(tokens, 3, 256, 17).cpu(), stable_hash(tokens.cpu(), 3, 256, 17))

@pytest.mark.gpu
def test_gpu_checkpoint_resume_equivalence(cfg, tmp_path):
    from src.training.engine import seed_all, amp
    seed_all(77); cfg['model']['moe_layers']=[]; model=LanguageModel(cfg).cuda(); opt,sched=make_optimizer(model,cfg)
    def step(m,o,s):
        x=torch.randint(0,512,(2,16),device='cuda'); o.zero_grad()
        with amp(cfg): loss=m(x,torch.roll(x,-1,1))['loss']
        loss.backward(); o.step(); s.step(); return x
    step(model,opt,sched); path=tmp_path/'gpu.pt'; save_training(path,model,opt,sched,{'cursor':2})
    expected_x=step(model,opt,sched); expected=copy.deepcopy(model.state_dict())
    restored=LanguageModel(cfg).cuda(); o2,s2=make_optimizer(restored,cfg); resume_training(path,restored,o2,s2,'cuda'); actual_x=step(restored,o2,s2)
    assert torch.equal(expected_x,actual_x)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in expected.items())

@pytest.mark.gpu
def test_mtp_greedy_verifier_parity(cfg):
    from src.eval.speculation import greedy,speculative_greedy
    cfg['mtp']['enabled']=True; cfg['model']['moe_layers']=[]; model=LanguageModel(cfg).cuda().eval(); prefix=torch.randint(0,512,(1,8),device='cuda')
    baseline,_=greedy(model,prefix,8,cfg); proposed,metrics=speculative_greedy(model,prefix,8,cfg)
    assert torch.equal(baseline,proposed) and 0<=metrics['accepted_length']<=3

@pytest.mark.gpu
def test_gpu_verification_in_no_grad():
    with torch.no_grad(): require_cuda('cuda')

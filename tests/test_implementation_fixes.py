import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.config import load_config
from src.model import LanguageModel
from src.model.layers import Attention
from src.memory.ngram import stable_hash
from src.training.engine import make_optimizer
from src.training.checkpoint import save_training, resume_training, rng_state
from src.training.data import ShardedTokens, PackedStream
from src.utils.hashing import sha256_file


def cfg():
    c = load_config('configs/test.yaml')
    c['memory']['enabled'] = True
    c['model']['moe_backend'] = 'grouped'
    c['training']['loss_chunk_tokens'] = 7
    return c


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_state_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_state_equal(x, y)
    else:
        assert a == b


def test_optimizer_groups_exhaustive_and_gate_bias_no_decay_drift():
    c = cfg(); m = LanguageModel(c); opt, _ = make_optimizer(m, c)
    names = {id(p): n for n, p in m.named_parameters()}
    intended = {n for n, p in m.named_parameters() if p.ndim < 2 or n.startswith('memory.tables.')}
    actual = {names[id(p)] for group in opt.param_groups if group['weight_decay'] == 0 for p in group['params']}
    assert actual == intended
    ids = [id(p) for group in opt.param_groups for p in group['params']]
    assert len(ids) == len(set(ids)) == len(names) and set(ids) == set(names)
    before = m.memory.gate.bias.detach().clone()
    m.memory.gate.bias.grad = torch.zeros_like(before)
    opt.step()
    assert torch.equal(before, m.memory.gate.bias)
    legacy = torch.optim.AdamW([m.memory.gate.bias], lr=.01, weight_decay=.1)
    m.memory.gate.bias.grad = torch.zeros_like(before); legacy.step()
    assert not torch.equal(before, m.memory.gate.bias)


def test_profiler_preserves_constant_lr_and_training_decay_policy():
    from tools.profile import profile_optimizer
    c = cfg(); m = LanguageModel(c)
    opt, schedule = profile_optimizer(m, c)
    training_opt, _ = make_optimizer(m, c)
    assert [g['param_names'] for g in opt.param_groups] == [g['param_names'] for g in training_opt.param_groups]
    assert [g['weight_decay'] for g in opt.param_groups] == [g['weight_decay'] for g in training_opt.param_groups]
    for _ in range(4):
        opt.step(); schedule.step()
        assert all(g['lr'] == c['training']['lr'] for g in opt.param_groups)
    c['training']['optimizer_decay_policy'] = 'legacy_all'
    legacy, _ = profile_optimizer(m, c)
    assert len(legacy.param_groups) == 1 and legacy.decay_policy == 'legacy_all'


@pytest.mark.parametrize('activation_checkpointing', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.gpu)])
def test_loss_only_forward_backward_parity(activation_checkpointing, device):
    c = cfg(); c['model']['activation_checkpointing'] = activation_checkpointing
    a = LanguageModel(c).to(device); b = LanguageModel(c).to(device); b.load_state_dict(a.state_dict())
    tokens = torch.randint(0, 512, (2, 15), device=device); target = torch.roll(tokens, -1, 1); target[0, 4] = -100
    with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
        full = a(tokens, target); minimal = b(tokens, target, return_outputs=False)
    assert set(minimal) == {'aux_loss', 'loss', 'lm_loss'}
    torch.testing.assert_close(full['loss'], minimal['loss'], atol=1e-6, rtol=1e-6)
    full['loss'].backward(); minimal['loss'].backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert (pa.grad is None) == (pb.grad is None)
        if pa.grad is not None: torch.testing.assert_close(pa.grad, pb.grad, atol=.002 if device=='cuda' else 2e-6, rtol=.05 if device=='cuda' else 2e-5)
    assert not b.memory.tables[0].weight.grad.is_sparse  # Deliberately retain dense Adam semantics.


def test_rope_cache_exact_reuse_lengths_and_dtype():
    m = Attention(cfg()['model'])
    def reference(x):
        positions = torch.arange(x.shape[-2], dtype=torch.float32, device=x.device)
        freq = m.theta ** (-torch.arange(0, m.hd, 2, dtype=torch.float32, device=x.device) / m.hd)
        phase = positions[:, None] * freq[None]
        a, b = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack((a*phase.cos()-b*phase.sin(), a*phase.sin()+b*phase.cos()), -1).flatten(-2).to(x.dtype)
    for length in (17, 4, 31, 31):
        x = torch.randn(2, 2, length, m.hd)
        assert torch.equal(m.rope(x), reference(x))
        pointer = m._rope_cos.data_ptr(); m.rope(x)
        assert pointer == m._rope_cos.data_ptr()
    m.bfloat16(); x = x.bfloat16()
    assert torch.equal(m.rope(x), reference(x))
    assert m._rope_cos.dtype == torch.float32 and not any('rope' in n for n in m.state_dict())


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.gpu)])
def test_full_resume_states_and_cpu_staging(tmp_path, monkeypatch, device):
    c = cfg(); m = LanguageModel(c).to(device); opt, scheduler = make_optimizer(m, c)
    torch.manual_seed(44); random.seed(44); np.random.seed(44)
    x = torch.randint(0, 512, (2, 8), device=device)
    m(x, x, return_outputs=False)['loss'].backward(); opt.step(); scheduler.step(); opt.zero_grad()
    path = tmp_path / 'resume.pt'
    save_training(path, m, opt, scheduler, dict(step=11, tokens_seen=1408, cursor=22))
    expected = copy.deepcopy((m.state_dict(), opt.state_dict(), scheduler.state_dict(), rng_state()))
    restored = LanguageModel(c).to(device); o2, s2 = make_optimizer(restored, c)
    real_load = torch.load; locations = []
    def checked_load(*args, **kwargs):
        locations.append(kwargs['map_location']); return real_load(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', checked_load)
    meta = resume_training(path, restored, o2, s2, device)
    assert locations == ['cpu'] and (meta['step'], meta['tokens_seen'], meta['cursor']) == (11, 1408, 22)
    assert_state_equal(expected, (restored.state_dict(), o2.state_dict(), s2.state_dict(), rng_state()))


def test_legacy_optimizer_group_migration_retains_moments_steps_scheduler(tmp_path):
    c = cfg(); legacy_cfg = copy.deepcopy(c); legacy_cfg['training']['optimizer_decay_policy'] = 'legacy_all'
    m = LanguageModel(legacy_cfg); opt, scheduler = make_optimizer(m, legacy_cfg)
    x = torch.randint(0, 512, (2, 8)); m(x, x)['loss'].backward(); opt.step(); scheduler.step()
    path = tmp_path / 'legacy.pt'; save_training(path, m, opt, scheduler, dict(step=1))
    new = LanguageModel(c); o2, s2 = make_optimizer(new, c)
    meta = resume_training(path, new, o2, s2, 'cpu')
    assert 'optimizer_group_migration' in meta
    assert s2.last_epoch == scheduler.last_epoch and s2.get_last_lr() == scheduler.get_last_lr() * 2
    for pa, pb in zip(m.parameters(), new.parameters()):
        assert torch.equal(pa, pb)
        assert_state_equal(opt.state.get(pa, {}), o2.state.get(pb, {}))
    assert o2.param_groups[1]['weight_decay'] == 0


def test_sharded_tokens_lru_empty_boundaries_and_escaping_copies(tmp_path):
    files = []
    for i, n in enumerate((0, 4, 0, 7, 3)):
        p = tmp_path / f'{i}.bin'; np.arange(sum((0,4,0,7,3)[:i]),sum((0,4,0,7,3)[:i])+n,dtype=np.uint16).tofile(p); files.append(p)
    tokens = ShardedTokens(files, max_open=1); expected = np.arange(14, dtype=np.uint16)
    saved = tokens[:4]
    for start, end in ((0,14), (4,11), (-6,None), (12,2), (99,105), (None,None), (-99,99)):
        assert np.array_equal(tokens[start:end], expected[start:end])
        assert len(tokens._cache) <= 1
    tokens.close(); assert saved.tolist() == [0,1,2,3]
    assert len(ShardedTokens([])) == 0
    with pytest.raises(ValueError): tokens[::2]


def test_document_isolation_masks_targets_attention_and_ngram(tmp_path):
    np.arange(16,dtype=np.uint16).tofile(tmp_path/'train.bin')
    (tmp_path/'manifest.json').write_text(json.dumps({'packing_policy':'document_isolated_v1'}))
    (tmp_path/'train_documents.jsonl').write_text('\n'.join(json.dumps({'start':s,'length':8}) for s in (0,8)))
    stream = PackedStream(tmp_path,'train',context=10,seed=42)
    x,y = stream.next(1,'cpu'); ids = stream.last_segment_ids
    assert y[0,7] == -100 and y[0,8] == 9
    assert stream.last_prediction_count == 9
    m = LanguageModel(cfg()).eval(); changed = x.clone(); changed[:,:8] = 99
    with torch.no_grad():
        a=m(x,segment_ids=ids)['logits']; b=m(changed,segment_ids=ids)['logits']
    torch.testing.assert_close(a[:,8:],b[:,8:],atol=0,rtol=0)
    assert torch.equal(stable_hash(x,3,256,17,ids)[:,8:], stable_hash(x[:,8:],3,256,17))
    cursor=stream.cursor; resumed=PackedStream(tmp_path,'train',10,42,cursor)
    assert all(torch.equal(a,b) for a,b in zip(stream.next(1,'cpu'),resumed.next(1,'cpu')))


def test_streaming_hash_avoids_read_bytes(tmp_path, monkeypatch):
    path=tmp_path/'blob'; path.write_bytes(b'x'*(3*1024**2+19))
    import hashlib
    expected=hashlib.sha256(b'x'*(3*1024**2+19)).hexdigest()
    monkeypatch.setattr(Path,'read_bytes',lambda p: (_ for _ in ()).throw(AssertionError('unbounded read')))
    assert sha256_file(path) == expected


@pytest.mark.gpu
def test_native_gqa_cuda_forward_backward_parity():
    c=cfg()['model']; a=Attention(c).cuda(); b=Attention(dict(c,gqa_backend='native')).cuda(); b.load_state_dict(a.state_dict())
    xa=torch.randn(2,33,c['d_model'],device='cuda',requires_grad=True); xb=xa.detach().clone().requires_grad_()
    with torch.autocast('cuda',dtype=torch.bfloat16): ya=a(xa); yb=b(xb)
    torch.testing.assert_close(ya,yb,atol=.002,rtol=.03)
    ya.square().mean().backward(); yb.square().mean().backward()
    torch.testing.assert_close(xa.grad,xb.grad,atol=.002,rtol=.05)
    for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa.grad,pb.grad,atol=.002,rtol=.05)

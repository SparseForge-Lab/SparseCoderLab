import pytest
import torch
from src.config import load_config
from src.memory.ngram import NgramMemory

def test_evaluation_ablation_zero_residual_and_restores():
    cfg = load_config('configs/test.yaml')
    memory = NgramMemory(cfg['memory'], 32)
    x = torch.randn(2, 8, 32); tokens = torch.randint(0, 512, (2, 8))
    memory.eval(); normal = memory(x, tokens)
    memory.ablate = True
    assert torch.equal(memory(x, tokens), x)
    memory.train()
    with pytest.raises(RuntimeError, match='evaluation-only'): memory(x, tokens)
    memory.eval(); memory.ablate = False
    assert torch.equal(memory(x, tokens), normal)

def test_table_diagnostics_identify_gradient_rows():
    cfg = load_config('configs/test.yaml'); memory = NgramMemory(cfg['memory'], 32)
    x = torch.randn(2, 8, 32); tokens = torch.randint(0, 512, (2, 8))
    memory(x, tokens).square().mean().backward()
    report = memory.diagnostics(tokens)
    assert 0 < report['gate_mean_last_microbatch'] < 1
    for table, bank in zip(memory.tables, report['table_gradients']):
        assert bank['gradient_norm_after_clip'] > 0
        assert bank['rows_with_gradient_this_step'] == int((table.weight.grad.abs().sum(-1) != 0).sum())

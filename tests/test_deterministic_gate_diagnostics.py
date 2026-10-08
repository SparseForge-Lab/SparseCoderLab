from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.eval.research import evaluate


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA diagnostic regression')
def test_ngram_gate_histogram_under_strict_cuda_determinism(tmp_path):
    # Exercise the real evaluator on a fixed CUDA gate vector. No model-weight
    # computation is needed to reproduce the CUDA float histc rejection.
    gates = [0., .1, .2, .5, .9999, 1.]

    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.memory = SimpleNamespace(ablate=False, statistics=lambda tokens: {'banks': []})

        def forward(self, tokens, targets, **kwargs):
            self.memory.last_gate = torch.tensor(gates, device=tokens.device)
            return {'lm_loss': torch.tensor(2., device=tokens.device)}

    np.arange(65, dtype=np.uint16).tofile(tmp_path/'val.bin')
    cfg = {'data': {'shards': str(tmp_path)}, 'training': {'context': 32, 'microbatch': 1, 'bf16': True}}
    index = {'mixed_batches': 1, 'minimum_documents': 1, 'minimum_predictions': 1, 'scope': 'fixture',
        'documents': {'code/Python': [{'start': 0, 'length': 7, 'sha256': 'fixture', 'language': 'Python'}]}}
    previous = torch.are_deterministic_algorithms_enabled()
    warning_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        result = evaluate(Fixture(), cfg, index)
        histogram = result['ngram']['code/Python']['gate_histogram10bins']
        assert histogram == torch.histc(torch.tensor(gates), bins=10, min=0, max=1).int().tolist()
        assert sum(histogram) == result['ngram']['code/Python']['tokens'] == len(gates)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warning_only)

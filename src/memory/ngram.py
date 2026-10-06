from __future__ import annotations
from collections import Counter
import torch
from torch import nn

def stable_hash(tokens: torch.Tensor, order: int, rows: int, seed: int) -> torch.Tensor:
    """Causal modular arithmetic; bounded intermediates <2**63 on every machine."""
    modulus = 2147483647
    out = torch.full_like(tokens, seed % modulus)
    for offset in range(order - 1, -1, -1):
        previous = torch.zeros_like(tokens)
        if offset == 0: previous = tokens + 1
        elif offset < tokens.shape[1]: previous[:, offset:] = tokens[:, :-offset] + 1
        out = (out * 65599 + previous) % modulus
    return out % rows

class NgramMemory(nn.Module):
    def __init__(self, cfg: dict, d_model: int):
        super().__init__(); self.cfg = cfg
        assert cfg['banks'] == len(cfg['orders']) == len(cfg['seeds'])
        self.tables = nn.ModuleList(nn.Embedding(cfg['rows'], cfg['dim']) for _ in cfg['orders'])
        self.projection = nn.Linear(cfg['banks'] * cfg['dim'], d_model, bias=False)
        self.gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, cfg['gate_bias'])
        self.ablate = False
        self.last_gate = None
    def bucket_ids(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        return [stable_hash(tokens, order, self.cfg['rows'], seed) for order, seed in zip(self.cfg['orders'], self.cfg['seeds'])]
    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if self.ablate:
            if self.training: raise RuntimeError('Memory ablation is evaluation-only')
            return x
        vectors = [table(ids) for table, ids in zip(self.tables, self.bucket_ids(tokens))]
        gate = torch.sigmoid(self.gate(x))
        self.last_gate = gate.detach()
        return x + gate * self.projection(torch.cat(vectors, -1))
    def diagnostics(self, tokens: torch.Tensor) -> dict:
        banks = []
        for table in self.tables:
            grad = table.weight.grad
            banks.append({'gradient_norm_after_clip': float(grad.norm()) if grad is not None else None,
                          'rows_with_gradient_this_step': int((grad.abs().sum(-1) != 0).sum()) if grad is not None else 0})
        return {'gate_mean_last_microbatch': float(self.last_gate.float().mean()),
                'gate_min_last_microbatch': float(self.last_gate.min()),
                'gate_max_last_microbatch': float(self.last_gate.max()),
                'table_gradients': banks, 'sample_statistics': self.statistics(tokens),
                'update_scope': 'Nonzero accumulated gradient rows sampled at log steps. AdamW also applies dense decay/momentum updates; not lifetime unique utilization.'}
    def statistics(self, tokens: torch.Tensor) -> dict:
        values = tokens.cpu().tolist(); banks = []
        for order, ids in zip(self.cfg['orders'], self.bucket_ids(tokens)):
            signatures: dict[int, set] = {}; counts = Counter(ids.cpu().flatten().tolist())
            for seq, buckets in zip(values, ids.cpu().tolist()):
                padded=[0]*(order-1)+[v+1 for v in seq]
                for t, bucket in enumerate(buckets):
                    signature = tuple(padded[t:t + order])
                    signatures.setdefault(bucket, set()).add(signature)
            distinct = sum(len(v) for v in signatures.values())
            banks.append({'order': order, 'utilized_rows': len(counts), 'rows': self.cfg['rows'],
                          'distinct_ngrams': distinct, 'collisions': distinct - len(signatures),
                          'collision_fraction': (distinct - len(signatures)) / max(distinct, 1),
                          'lookup_frequency_histogram': dict(sorted(Counter(counts.values()).items())),
                          'bucket_utilization_histogram': dict(sorted(Counter(map(len, signatures.values())).items()))})
        return {'banks': banks, 'scope': 'Only the supplied bounded sample; not whole-corpus collision estimates'}

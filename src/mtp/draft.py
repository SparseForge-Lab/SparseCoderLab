from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from src.model.layers import RMSNorm, SwiGLU

def horizon_targets(tokens: torch.Tensor, horizon: int) -> torch.Tensor:
    if horizon < 1: raise ValueError('horizon is one-based')
    return tokens[:, horizon:]

class DraftBlock(nn.Module):
    def __init__(self, dim: int, ffn: int, eps: float):
        super().__init__(); self.norm = RMSNorm(dim, eps); self.mix = nn.Linear(2 * dim, dim, bias=False); self.ff = SwiGLU(dim, ffn)
    def forward(self, state: torch.Tensor, previous_embedding: torch.Tensor) -> torch.Tensor:
        state = state + self.mix(torch.cat((self.norm(state), previous_embedding), -1))
        return state + self.ff(self.norm(state))

class MultiTokenDraft(nn.Module):
    def __init__(self, cfg: dict, model_cfg: dict):
        super().__init__(); self.cfg = cfg; self.horizons = cfg['horizons']
        assert self.horizons == 3
        assert cfg['design'] in ('shared_mtp_3step', 'three_layer_mtp')
        n = 1 if cfg['design'] == 'shared_mtp_3step' else self.horizons
        self.blocks = nn.ModuleList(DraftBlock(model_cfg['d_model'], cfg['ffn'], model_cfg['norm_eps']) for _ in range(n))
    def block(self, h: int) -> DraftBlock: return self.blocks[0 if len(self.blocks) == 1 else h]
    def losses(self, hidden: torch.Tensor, tokens: torch.Tensor, embedding: nn.Embedding) -> tuple[torch.Tensor, dict]:
        state = hidden; total = hidden.new_zeros(()); metrics = {}
        for h in range(1, self.horizons + 1):
            length = tokens.shape[1] - h
            if length <= 0: continue
            state = self.block(h - 1)(state[:, :length], embedding(tokens[:, h - 1:h - 1 + length]))
            logits = F.linear(state, embedding.weight).float(); target = horizon_targets(tokens, h)
            loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
            total = total + self.cfg['loss_weights'][h - 1] * loss
            metrics[f't+{h}_loss'] = float(loss.detach()); metrics[f't+{h}_accuracy'] = float((logits.argmax(-1) == target).float().mean().detach())
        return total, metrics
    @torch.no_grad()
    def generate(self, hidden: torch.Tensor, previous_token: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        state = hidden; out = []
        for h in range(self.horizons):
            state = self.block(h)(state, embedding(previous_token)); previous_token = F.linear(state, embedding.weight).argmax(-1)
            out.append(previous_token)
        return torch.stack(out, -1)

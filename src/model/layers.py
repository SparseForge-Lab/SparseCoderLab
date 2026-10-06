from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate: int):
        super().__init__(); self.up = nn.Linear(dim, 2 * intermediate, bias=False); self.down = nn.Linear(intermediate, dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.up(x).chunk(2, dim=-1); return self.down(F.silu(a) * b)

class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__(); self.qh = cfg['q_heads']; self.kh = cfg['kv_heads']; self.hd = cfg['head_dim']; self.theta = cfg['rope_theta']
        dim = cfg['d_model']; self.q = nn.Linear(dim, self.qh * self.hd, bias=False)
        self.kv = nn.Linear(dim, 2 * self.kh * self.hd, bias=False); self.out = nn.Linear(dim, dim, bias=False)
    def rope(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        freq = self.theta ** (-torch.arange(0, self.hd, 2, device=x.device, dtype=torch.float32) / self.hd)
        phase = positions[:, None] * freq[None, :]
        a, b = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack((a * phase.cos() - b * phase.sin(), a * phase.sin() + b * phase.cos()), -1).flatten(-2).to(x.dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.qh, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(b, t, 2, self.kh, self.hd).unbind(2)
        k = self.rope(k.transpose(1, 2)); q = self.rope(q); v = v.transpose(1, 2)
        # Identical native SDPA path for every architecture, explicit GQA expansion.
        k = k.repeat_interleave(self.qh // self.kh, 1); v = v.repeat_interleave(self.qh // self.kh, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, -1))

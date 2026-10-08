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
        self.gqa_backend = cfg.get('gqa_backend', 'repeat')
        if self.gqa_backend not in ('repeat', 'native'):
            raise ValueError('Unknown GQA backend')
        dim = cfg['d_model']; self.q = nn.Linear(dim, self.qh * self.hd, bias=False)
        self.kv = nn.Linear(dim, 2 * self.kh * self.hd, bias=False); self.out = nn.Linear(dim, dim, bias=False)
        self.register_buffer('_rope_cos', torch.empty(0), persistent=False)
        self.register_buffer('_rope_sin', torch.empty(0), persistent=False)
    def rope(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-2]
        if (self._rope_cos.device != x.device or self._rope_cos.dtype != torch.float32
                or self._rope_cos.shape[0] < length):
            positions = torch.arange(length, device=x.device, dtype=torch.float32)
            freq = self.theta ** (-torch.arange(0, self.hd, 2, device=x.device, dtype=torch.float32) / self.hd)
            phase = positions[:, None] * freq[None, :]
            self._rope_cos, self._rope_sin = phase.cos(), phase.sin()
        cosine, sine = self._rope_cos[:length], self._rope_sin[:length]
        a, b = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).flatten(-2).to(x.dtype)
    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.qh, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(b, t, 2, self.kh, self.hd).unbind(2)
        k = self.rope(k.transpose(1, 2)); q = self.rope(q); v = v.transpose(1, 2)
        # Preserve the historical expansion unless native GQA is requested.
        if self.gqa_backend == 'repeat':
            k = k.repeat_interleave(self.qh // self.kh, 1); v = v.repeat_interleave(self.qh // self.kh, 1)
        mask = None
        if segment_ids is not None:
            if segment_ids.shape != (b, t):
                raise ValueError('Document segment shape mismatch')
            causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            mask = (segment_ids[:, :, None] == segment_ids[:, None, :])[:, None] & causal
        y = F.scaled_dot_product_attention(q, k, v, is_causal=mask is None, attn_mask=mask, dropout_p=0.0,
                                          enable_gqa=self.gqa_backend == 'native')
        return self.out(y.transpose(1, 2).contiguous().view(b, t, -1))

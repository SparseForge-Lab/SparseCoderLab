"""Preserve reference AdamW semantics for empty experts after stacked execution."""
from __future__ import annotations
import torch
from src.moe.router import FreeMoE

class GroupedAdamW(torch.optim.AdamW):
    def __init__(self, model, param_groups=None, **kwargs):
        super().__init__(model.parameters() if param_groups is None else param_groups,**kwargs)
        self.grouped_modules=[m for m in model.modules() if isinstance(m,FreeMoE) and m.backend=='grouped']
    def zero_grad(self,set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for m in self.grouped_modules: m.active_since_zero.zero_()
    def step(self,closure=None):
        if closure is not None: raise ValueError('GroupedAdamW requires explicit forward/backward, no optimizer closure')
        # Dispatch now returns None for inactive expert parameters directly.
        # Accumulation retains earlier gradients naturally; no host sync needed.
        return super().step()

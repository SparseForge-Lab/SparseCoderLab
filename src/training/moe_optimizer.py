"""Preserve reference AdamW semantics for empty experts after stacked execution."""
from __future__ import annotations
import torch
from src.moe.router import FreeMoE

class GroupedAdamW(torch.optim.AdamW):
    def __init__(self, model, **kwargs):
        super().__init__(model.parameters(),**kwargs)
        self.grouped_modules=[m for m in model.modules() if isinstance(m,FreeMoE) and m.backend=='grouped']
    def zero_grad(self,set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for m in self.grouped_modules: m.active_since_zero.zero_()
    def step(self,closure=None):
        if closure is not None: raise ValueError('GroupedAdamW requires explicit forward/backward, no optimizer closure')
        # StackBackward returns zero gradients for empty experts. Reference gives
        # None, skipping decay/momentum/step counters. Restore that exact behavior
        # over the union of all accumulation microbatches before ordinary AdamW.
        for m in self.grouped_modules:
            for active,expert in zip(m.active_since_zero.cpu().tolist(),m.experts):
                if not active:
                    for parameter in expert.parameters(): parameter.grad=None
        return super().step()

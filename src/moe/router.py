from __future__ import annotations
import torch
from torch import nn
from src.model.layers import SwiGLU

class FreeMoE(nn.Module):
    """No capacity drops, no page constraints; token dispatch reference implementation."""
    def __init__(self, cfg: dict):
        super().__init__(); self.k = cfg['top_k']; self.n = cfg['experts']
        self.router = nn.Linear(cfg['d_model'], self.n, bias=False)
        self.experts = nn.ModuleList(SwiGLU(cfg['d_model'], cfg['expert_ffn']) for _ in range(self.n))
        self.backend = cfg.get('moe_backend', 'reference')
        if self.backend not in ('reference','grouped'): raise ValueError('Unknown MoE backend')
        if self.backend == 'grouped' and self.k != 1: raise ValueError('Grouped backend is Top1 only; use reference for Top2')
        self.register_buffer('active_since_zero', torch.zeros(self.n,dtype=torch.bool), persistent=False)
        self.last: dict = {}
        if self.backend == 'grouped':
            self.register_buffer('_expert_up', torch.empty(0), persistent=False)
            self.register_buffer('_expert_down', torch.empty(0), persistent=False)
            self._rebuild_banks()
    def _ensure_banks(self):
        for name in ('up','down'):
            bank=getattr(self,'_expert_'+name)
            stride=bank[0].numel()*bank.element_size()
            if any(getattr(expert,name).weight.data_ptr()!=bank.data_ptr()+i*stride
                   for i,expert in enumerate(self.experts)):
                self._rebuild_banks(); break
    def _rebuild_banks(self):
        with torch.no_grad():
            for name in ('up', 'down'):
                parameters = [getattr(expert, name).weight for expert in self.experts]
                bank = torch.stack(parameters).detach()
                setattr(self, '_expert_' + name, bank)
                for parameter, view in zip(parameters, bank.unbind(0)):
                    parameter.data = view
    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self.backend == 'grouped':
            self._rebuild_banks()
        return result
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape; flat = x.reshape(-1, shape[-1]); probs = self.router(flat).float().softmax(-1)
        weights, indices = probs.topk(self.k, dim=-1)
        # Top1 retains selected probability so LM gradients reach the router; Top2 normalizes.
        if self.k > 1: weights = weights / weights.sum(-1, keepdim=True)
        if self.backend == 'reference':
            output = torch.zeros_like(flat)
            for expert_id, expert in enumerate(self.experts):
                token, slot = torch.where(indices == expert_id)
                if token.numel():
                    values = expert(flat[token]) * weights[token, slot, None].to(flat.dtype)
                    output.index_add_(0, token, values.to(output.dtype))
            counts = torch.bincount(indices.flatten(), minlength=self.n).float()
            capacity = None
        else:
            from src.moe.grouped import dispatch
            self._ensure_banks()
            integer_counts=torch.zeros(self.n,dtype=torch.long,device=flat.device).scatter_add_(0,indices[:,0],torch.ones(flat.shape[0],dtype=torch.long,device=flat.device))
            output,capacity=dispatch(flat,indices,weights,self.experts,integer_counts,self._expert_up,self._expert_down)
            counts=integer_counts.float()
            if self.training and torch.is_grad_enabled(): self.active_since_zero.logical_or_(integer_counts>0)
        load = counts / indices.numel()
        aux = self.n * (load.detach() * probs.mean(0)).sum()
        self.last = {'experts': indices.detach().view(*shape[:-1], self.k),
                     'probabilities': probs.detach().view(*shape[:-1], self.n),
                     'selected_weights': weights.detach().view(*shape[:-1], self.k),
                     'entropy': -(probs.detach() * probs.detach().clamp_min(1e-9).log()).sum(-1).mean(),
                     'load': counts.detach(), 'imbalance': counts.std(unbiased=False) / counts.mean().clamp_min(1),
                     'backend': self.backend, 'padded_capacity': capacity}
        return output.view(shape), aux

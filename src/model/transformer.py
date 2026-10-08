from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from src.model.layers import Attention, RMSNorm, SwiGLU
from src.moe.router import FreeMoE
from src.memory.ngram import NgramMemory
from src.mtp.draft import MultiTokenDraft
from src.model.loss import linear_cross_entropy

class Block(nn.Module):
    def __init__(self, cfg: dict, capacity: bool):
        super().__init__(); d = cfg['d_model']; self.an = RMSNorm(d, cfg['norm_eps']); self.fn = RMSNorm(d, cfg['norm_eps'])
        self.attention = Attention(cfg); self.resident = SwiGLU(d, cfg['ffn']); self.moe = FreeMoE(cfg) if capacity else None
    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.an(x), segment_ids); normalized = self.fn(x); residual = self.resident(normalized)
        aux = x.new_zeros(())
        if self.moe is not None:
            routed, aux = self.moe(normalized); residual = residual + routed
        return x + residual, aux

class LanguageModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__(); self.cfg = cfg; m = cfg['model']
        self.embedding = nn.Embedding(m['vocab_size'], m['d_model'])
        self.blocks = nn.ModuleList(Block(m, i in m['moe_layers']) for i in range(m['layers']))
        self.norm = RMSNorm(m['d_model'], m['norm_eps'])
        self.memory = NgramMemory(cfg['memory'], m['d_model']) if cfg['memory']['enabled'] else None
        self.mtp = MultiTokenDraft(cfg['mtp'], m) if cfg['mtp']['enabled'] else None
        self.apply(self.initialize)
        if self.memory:
            nn.init.zeros_(self.memory.gate.weight); nn.init.constant_(self.memory.gate.bias, cfg['memory']['gate_bias'])
    def initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.cfg['model']['init_std'])
            if isinstance(module, nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)
    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None, *, mtp_tokens: torch.Tensor | None = None,
                return_outputs: bool = True, segment_ids: torch.Tensor | None = None) -> dict:
        if targets is None and not return_outputs:
            raise ValueError('Loss-only forward requires targets')
        if segment_ids is not None and self.mtp is not None and mtp_tokens is not None:
            raise ValueError('Document isolation with MTP requires boundary-aware horizon targets')
        x = self.embedding(tokens); aux = x.new_zeros(())
        for i, block in enumerate(self.blocks):
            if self.memory is not None and i == self.cfg['memory']['layer']: x = self.memory(x, tokens, segment_ids=segment_ids)
            if self.cfg['model']['activation_checkpointing'] and self.training:
                x, loss = checkpoint(block, x, segment_ids, use_reentrant=False)
            else: x, loss = block(x, segment_ids)
            aux = aux + loss
        hidden = self.norm(x)
        result = {'aux_loss': aux}
        if return_outputs:
            logits = F.linear(hidden, self.embedding.weight)
            result.update(logits=logits, hidden=hidden)
        if targets is not None:
            lm_loss = (F.cross_entropy(logits.float().flatten(0, 1), targets.flatten()) if return_outputs else
                       linear_cross_entropy(hidden, self.embedding.weight, targets,
                                            self.cfg['training'].get('loss_chunk_tokens', 1024)))
            result.update(lm_loss=lm_loss, loss=lm_loss + self.cfg['model']['aux_weight'] * aux)
            if self.mtp is not None and mtp_tokens is not None:
                mtp_loss, metrics = self.mtp.losses(hidden, mtp_tokens, self.embedding)
                result['loss'] = result['loss'] + mtp_loss; result['mtp_metrics'] = metrics
        return result
    def routes(self) -> dict[int, dict]:
        return {i: block.moe.last for i, block in enumerate(self.blocks) if block.moe is not None}

from __future__ import annotations
import time
import torch
from src.training.engine import amp

@torch.no_grad()
def greedy(model, prefix: torch.Tensor, tokens: int, cfg: dict) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize(); start = time.perf_counter(); sequence = prefix.clone()
    for _ in range(tokens):
        with amp(cfg): next_token = model(sequence)['logits'][:, -1].argmax(-1)
        sequence = torch.cat((sequence, next_token[:, None]), 1)
    torch.cuda.synchronize(); return sequence, time.perf_counter() - start

@torch.no_grad()
def speculative_greedy(model, prefix: torch.Tensor, tokens: int, cfg: dict) -> tuple[torch.Tensor, dict]:
    if model.mtp is None: raise ValueError('MTP required')
    if prefix.shape[0] != 1: raise ValueError('Reference verifier uses batch=1')
    torch.cuda.synchronize(); start = time.perf_counter(); sequence = prefix.clone(); accepted = []; draft_seconds = 0.0
    while sequence.shape[1] < prefix.shape[1] + tokens:
        remaining = prefix.shape[1] + tokens - sequence.shape[1]
        with amp(cfg):
            base = model(sequence); torch.cuda.synchronize(); draft_start = time.perf_counter()
            proposals = model.mtp.generate(base['hidden'][:, -1], sequence[:, -1], model.embedding)[:, :min(3, remaining)]
            torch.cuda.synchronize(); draft_seconds += time.perf_counter() - draft_start
            verification = model(torch.cat((sequence, proposals), 1))['logits']
        offset = sequence.shape[1] - 1; count = 0
        for h in range(proposals.shape[1]):
            actual = verification[:, offset + h].argmax(-1)
            if int(actual) != int(proposals[:, h]):
                sequence = torch.cat((sequence, proposals[:, :h], actual[:, None]), 1); break
            count += 1
        else: sequence = torch.cat((sequence, proposals), 1)
        accepted.append(count)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
    return sequence, {'accepted_length': sum(accepted) / max(len(accepted), 1), 'accepted_per_iteration': accepted,
                      'tokens_s_including_draft': tokens / elapsed, 'wall_seconds': elapsed, 'draft_seconds': draft_seconds,
                      'method': 'Exact greedy parity; full-prefix SDPA reference without KV cache. Not stochastic speculative sampling.'}

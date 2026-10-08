"""Bounded-logit causal cross entropy with recomputation during backward."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def linear_cross_entropy(hidden, weight, targets, chunk_tokens=256):
    if chunk_tokens < 1:
        raise ValueError('Loss chunk size must be positive')
    hidden = hidden.flatten(0, -2)
    targets = targets.flatten()

    def chunk_loss(values, table, labels):
        return F.cross_entropy(F.linear(values, table).float(), labels, reduction='sum')

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk_tokens):
        values, labels = hidden[start:start + chunk_tokens], targets[start:start + chunk_tokens]
        if torch.is_grad_enabled():
            value = checkpoint(chunk_loss, values, weight, labels, use_reentrant=False)
        else:
            value = chunk_loss(values, weight, labels)
        total = total + value
    # Same default ignore_index and all-ignored NaN behavior as ordinary CE.
    return total / (targets != -100).sum()

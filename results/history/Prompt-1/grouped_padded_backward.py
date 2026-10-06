"""Sorted, lossless Top1 dispatch using ordinary PyTorch batched GEMMs."""
from __future__ import annotations
import torch
from torch.nn import functional as F

def dispatch(flat, indices, weights, experts, counts):
    n=len(experts); tokens=flat.shape[0]
    # One bounded host read for allocation. No fixed capacity, overflow or drops.
    capacity=((int(counts.max().item())+15)//16)*16
    ids=indices[:,0]; order=torch.argsort(ids,stable=True)
    starts=counts.cumsum(0)-counts
    ranks=torch.arange(tokens,device=flat.device)-starts[ids[order]]
    slots=ids[order]*capacity+ranks
    packed=flat.new_zeros(n*capacity,flat.shape[1]).index_copy(0,slots,flat[order]).view(n,capacity,-1)
    # Registered parameters stay separate; temporary stacks preserve state_dict
    # names/optimizer IDs and remove per-expert forward dispatch.
    up=torch.stack([e.up.weight for e in experts]).transpose(1,2)
    down=torch.stack([e.down.weight for e in experts]).transpose(1,2)
    a,b=torch.bmm(packed,up).chunk(2,dim=-1)
    values=torch.bmm(F.silu(a)*b,down).flatten(0,1)[slots]
    values=values.to(flat.dtype)*weights[order,0,None].to(flat.dtype)
    output=torch.empty_like(flat).index_copy(0,order,values)
    return output,capacity

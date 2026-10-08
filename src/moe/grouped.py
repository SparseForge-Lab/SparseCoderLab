"""Sorted, lossless Top1 dispatch using ordinary PyTorch batched GEMMs."""
from __future__ import annotations
import torch
from torch.nn import functional as F

class _BatchedLinear(torch.autograd.Function):
    """Batched input gradients, unpadded weight reductions like the oracle.

    Zero padding changes GEMM reduction geometry by a few rounding units.
    At a trained Top1 near-tie those units can change later routing. Keeping
    weight reductions at each expert's true size avoids that amplification.
    """
    @staticmethod
    def forward(ctx,x,weight,sizes,*parameters):
        out=torch.bmm(x,weight.transpose(1,2))
        ctx.save_for_backward(x.to(out.dtype),weight.to(out.dtype))
        ctx.sizes=sizes
        return out
    @staticmethod
    def backward(ctx,grad):
        x,weight=ctx.saved_tensors
        dx=torch.bmm(grad,weight)
        dw=tuple(grad[i,:size].transpose(0,1) @ x[i,:size] if size else None
                 for i,size in enumerate(ctx.sizes))
        # The bank is storage only. Gradients belong to the original registered
        # expert Parameters, preserving names, Adam state and inactive None.
        return (dx,None,None,*dw)

def dispatch(flat, indices, weights, experts, counts, up, down):
    n=len(experts); tokens=flat.shape[0]
    # One bounded host read for allocation. No fixed capacity, overflow or drops.
    sizes=counts.cpu().tolist()
    capacity=((max(sizes)+15)//16)*16
    ids=indices[:,0]; order=torch.argsort(ids,stable=True)
    starts=counts.cumsum(0)-counts
    ranks=torch.arange(tokens,device=flat.device)-starts[ids[order]]
    slots=ids[order]*capacity+ranks
    packed=flat.new_zeros(n*capacity,flat.shape[1]).index_copy(0,slots,flat[order]).view(n,capacity,-1)
    a,b=_BatchedLinear.apply(packed,up,sizes,*(e.up.weight for e in experts)).chunk(2,dim=-1)
    values=_BatchedLinear.apply(F.silu(a)*b,down,sizes,*(e.down.weight for e in experts)).flatten(0,1)[slots]
    values=values.to(flat.dtype)*weights[order,0,None].to(flat.dtype)
    output=torch.empty_like(flat).index_copy(0,order,values)
    return output,capacity

"""Require CUDA, SM120 wheel coverage, CUDA >=12.8 and real BF16 SDPA backward."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F

def verify() -> dict:
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; CPU fallback forbidden')
    cc = torch.cuda.get_device_capability()
    arches = torch.cuda.get_arch_list()
    if cc == (12, 0) and 'sm_120' not in arches: raise RuntimeError('Wheel lacks sm_120')
    if not torch.version.cuda or tuple(map(int, torch.version.cuda.split('.')[:2])) < (12, 8):
        raise RuntimeError('CUDA runtime >=12.8 required')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unsupported')
    # Verification is also called from no_grad evaluation/trace collectors.
    with torch.enable_grad():
        x = torch.randn(2, 5, 32, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        y = F.scaled_dot_product_attention(x, x, x, is_causal=True)
        y.float().square().mean().backward(); torch.cuda.synchronize()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    return {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'compute_capability': cc, 'arch_list': arches, 'bf16_sdpa_backward': True,
            'tf32_supported': cc[0] >= 8, 'cpu_fallback': False}

if __name__ == '__main__':
    r = verify(); Path('results').mkdir(exist_ok=True)
    Path('results/gpu_verification.json').write_text(json.dumps(r, indent=2)); print(json.dumps(r, indent=2))

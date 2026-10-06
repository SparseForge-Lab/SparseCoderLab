from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml

def project(c: dict) -> dict:
    d = c['d_model']; layers = c['layers']; capacity = c['capacity_layers']; experts = c['experts']; k = c['top_k']
    if not 1 <= k <= experts or capacity > layers: raise ValueError('Invalid topology')
    attention = layers * (d * (c['q_heads'] + 2 * c['kv_heads']) * c['head_dim'] + d*d)
    resident = c['vocab_size'] * d + attention + layers * (3 * d * c['resident_ffn'] + 2 * d) + d
    routers = capacity * d * experts; expert_params = 3 * d * c['expert_ffn']; routed = capacity * experts * expert_params
    memory = c['ngram_params']; stored = resident + routers + routed + memory
    active = resident + routers + capacity * k * expert_params + c.get('ngram_touched', 0)
    bpp = {'bf16': 2, 'fp8': 1, 'int8': 1, 'int4': .5}[c['precision']]
    expert_bytes = expert_params * bpp; page_bytes = c['page_experts'] * expert_bytes
    # Distinct selected expert pages can overfetch; conservatively one whole page per selected expert.
    worst = capacity * k * page_bytes; cached = worst * (1 - c['cache_hit_rate'])
    resident_bytes = (resident + routers) * bpp; available = max(c['gpu_gib'] * 1024**3 - resident_bytes - c['activation_reserve_gib'] * 1024**3, 0)
    return {'stored_params_est': stored, 'active_params_est': active,
            'weight_bytes': {'bf16': stored*2, 'fp8': stored, 'int8': stored, 'int4': stored/2},
            'routed_expert_params': routed, 'resident_params': resident + routers, 'ngram_params': memory,
            'expert_bytes': expert_bytes, 'expert_page_bytes': page_bytes, 'worst_case_bytes_per_token': worst,
            'cached_bytes_per_token': cached, 'ssd_bandwidth_ceiling_tokens_s': c['ssd_mib_s'] * 1024**2 / cached if cached else None,
            'ram_requirement_bytes_all_weights': stored * bpp, 'ngram_host_bytes': memory * bpp,
            'fits_host_ram_all_weights': stored * bpp <= c['ram_gib'] * 1024**3,
            'vram_expert_cache_bytes_available': available, 'vram_expert_cache_coverage': min(available / max(routed*bpp, 1), 1),
            'caveats': 'Assumption-driven estimates; quantization scales/metadata, allocator, KV/cache traffic and SSD latency excluded. Page overfetch conservative. Cache hit rate must be measured at the projected topology; micro hit rates do not transfer.'}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/large_candidate.yaml')
    for key in ['d_model','layers','capacity_layers','experts','expert_ffn','top_k','resident_ffn','ngram_params','page_experts']:
        p.add_argument('--' + key.replace('_','-'), type=int)
    for key in ['gpu_gib','ram_gib','ssd_mib_s','cache_hit_rate']: p.add_argument('--' + key.replace('_','-'), type=float)
    p.add_argument('--precision', choices=['bf16','fp8','int8','int4']); a = p.parse_args(); cfg = yaml.safe_load(Path(a.config).read_text())
    for k, v in vars(a).items():
        if k != 'config' and v is not None: cfg[k] = v
    report = {}
    for memory in [cfg['ngram_params']] if a.ngram_params is not None else cfg['memory_sweep']:
        for top_k in [cfg['top_k']] if a.top_k is not None else [1,2]: report[f'ngram={memory}/top_k={top_k}'] = project(dict(cfg, ngram_params=memory, top_k=top_k))
    Path('results/large_projection.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))

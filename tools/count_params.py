from __future__ import annotations
import argparse, json
from src.config import load_config
from src.model import LanguageModel

def count_model(model: LanguageModel) -> dict:
    groups = {'routed_experts': 0, 'router': 0, 'conditional_memory': 0, 'mtp': 0, 'resident': 0}
    for name, p in model.named_parameters():
        key = 'routed_experts' if '.moe.experts.' in name else 'router' if '.moe.router.' in name else 'conditional_memory' if name.startswith('memory.') else 'mtp' if name.startswith('mtp.') else 'resident'
        groups[key] += p.numel()
    m = model.cfg['model']; mem = model.cfg['memory']
    total = sum(groups.values())
    active = total - groups['routed_experts'] + groups['routed_experts'] * m['top_k'] // m['experts']
    if model.memory:
        active -= mem['banks'] * mem['rows'] * mem['dim']; active += mem['banks'] * mem['dim']
    return {'total': total, **groups, 'memory_tables': mem['banks'] * mem['rows'] * mem['dim'] if model.memory else 0,
            'active_estimate': active, 'base_active_without_mtp': active - groups['mtp'],
            'caveat': 'Tied embedding counted once; full vocab LM head touched. Active estimate is parameter access, not measured FLOPs. MTP counted once, although shared block unrolled three times.'}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--configs', nargs='+', default=['configs/dense_compute.yaml','configs/dense_size.yaml','configs/sparse_v3.yaml','configs/sparse_memory.yaml','configs/sparse_mtp.yaml','configs/sparse_top2.yaml'])
    a = p.parse_args(); report = {}
    for path in a.configs:
        cfg = load_config(path); report[cfg['name']] = count_model(LanguageModel(cfg))
    from pathlib import Path
    Path('results/parameter_counts.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))

from __future__ import annotations
from collections import Counter, defaultdict
from itertools import combinations
from src.moe.trace import global_experts

def expert_atlas(records: list[dict], experts_per_layer: int) -> dict:
    load = Counter(); edges = Counter(); usage = defaultdict(Counter); previous = None; entropy = []
    for record in records:
        current = global_experts(record, experts_per_layer); load.update(current)
        usage[f"{record.get('kind', 'unknown')}/{record.get('language', 'unknown')}"] .update(current)
        for a, b in combinations(set(current), 2): edges[tuple(sorted((a, b)))] += 1
        if previous is not None:
            for a, b in zip(previous, current):
                if a != b: edges[tuple(sorted((a, b)))] += 1
        previous = current
        if 'entropy' in record: entropy.extend(record['entropy'].values())
    return {'load': dict(load), 'edges': [{'a': a, 'b': b, 'weight': w} for (a, b), w in edges.items()],
            'conditioned_usage': {k: dict(v) for k, v in usage.items()},
            'router_entropy': sum(entropy) / max(len(entropy), 1),
            'edge_definition': 'Token co-activation + successive-token transitions (global IDs include physical layer).'}

def pack_experts(atlas: dict, page_size: int, graph: bool = True) -> dict[int, int]:
    if page_size < 1: raise ValueError('page_size must be positive')
    load = {int(k): v for k, v in atlas['load'].items()}; remaining = set(load); edges = Counter()
    for edge in atlas['edges']: edges[tuple(sorted((edge['a'], edge['b'])))] = edge['weight']
    assignment = {}; page = 0
    while remaining:
        first = max(remaining, key=lambda e: (load[e], -e)) if graph else min(remaining)
        group = [first]; remaining.remove(first)
        while remaining and len(group) < page_size:
            candidate = max(remaining, key=lambda e: (sum(edges[tuple(sorted((e, g)))] for g in group), load[e], -e)) if graph else min(remaining)
            group.append(candidate); remaining.remove(candidate)
        for expert in group: assignment[expert] = page
        page += 1
    return assignment

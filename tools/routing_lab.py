from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from src.config import load_config
from src.model import LanguageModel
from src.training.data import PackedStream
from src.training.engine import require_cuda, amp
from src.moe.trace import write_trace, read_trace, global_experts
from src.moe.atlas import expert_atlas, pack_experts
from src.runtime.cache import IOConfig, simulate

@torch.no_grad()
def collect(cfg: dict, checkpoint: Path, output: Path, max_tokens: int) -> None:
    require_cuda('cuda'); model = LanguageModel(cfg).cuda().eval()
    state = torch.load(checkpoint, map_location='cuda', weights_only=False); model.load_state_dict(state['model'])
    stream = PackedStream(Path(cfg['data']['shards']), 'val', cfg['training']['context'], cfg['data']['seed'])
    documents = [json.loads(line) for line in (Path(cfg['data']['shards']) / 'val_documents.jsonl').read_text().splitlines()]
    def records():
        count = 0
        while count < max_tokens:
            cursor = stream.cursor; x, _ = stream.next(1, 'cuda')
            epoch, pos = divmod(cursor, stream.count); offset = int(stream.order[pos]) * stream.context
            with amp(cfg): model(x)
            routes = model.routes()
            for t in range(x.shape[1]):
                doc = next((d for d in documents if d['start'] <= offset + t < d['start'] + d['length']), {})
                yield {'token': count, 'layers': {str(i): r['experts'][0, t].tolist() for i, r in routes.items()},
                       'probabilities': {str(i): r['probabilities'][0, t].tolist() for i, r in routes.items()},
                       'entropy': {str(i): float(-(r['probabilities'][0, t] * r['probabilities'][0, t].clamp_min(1e-9).log()).sum()) for i, r in routes.items()},
                       'kind': doc.get('kind', 'unknown'), 'language': doc.get('language', 'unknown')}
                count += 1
                if count >= max_tokens: break
    write_trace(output, records())

def analyze(cfg: dict, trace: Path, ssd_mib_s: float | None = None, ssd_latency_ms: float | None = None) -> dict:
    records = list(read_trace(trace)); e = cfg['model']['experts']; runtime = cfg['runtime']; cut = max(len(records) // 2, 1)
    fit, heldout = records[:cut], records[cut:]
    if not heldout: raise ValueError('Need held-out trace portion')
    atlas = expert_atlas(fit, e)
    # Include experts first observed in held-out data without fitting their affinities.
    for record in heldout:
        for expert in global_experts(record, e): atlas['load'].setdefault(expert, 0)
    io = IOConfig(runtime['expert_bytes'], runtime['vram_pages'] * runtime['expert_bytes'], runtime['ram_pages'] * runtime['expert_bytes'],
                  (ssd_mib_s or runtime['ssd_mib_s']) * 1024**2, (ssd_latency_ms if ssd_latency_ms is not None else runtime['ssd_latency_ms']) / 1000,
                  runtime['ram_mib_s'] * 1024**2, runtime['ram_latency_ms'] / 1000)
    experiments = []
    for size in runtime['page_sizes']:
        for graph in (False, True):
            mapping = pack_experts(atlas, size, graph)
            frequency = {}
            for record in fit:
                for expert in global_experts(record, e): frequency[mapping[expert]] = frequency.get(mapping[expert], 0) + 1
            pin_count = runtime['vram_pages'] // size // 2
            pinned = set(sorted(frequency, key=lambda p: (-frequency[p], p))[:pin_count])
            for policy in ('lru', 'lfu', 'frequency_pin'):
                result = simulate(heldout, mapping, e, io, 'lfu' if policy == 'lfu' else 'lru', pinned if policy == 'frequency_pin' else None)
                experiments.append(dict(page_size=size, graph_packed=graph, policy=policy, **result))
            # Task profile is fitted on prefix only. Current task label assumed available.
            from collections import Counter
            profiles = {}
            for record in fit: profiles.setdefault(record.get('kind', 'unknown'), Counter()).update(mapping[x] for x in global_experts(record, e))
            predictions = {i: [p for p, _ in profiles.get(record.get('kind', 'unknown'), Counter()).most_common(max(pin_count, 1))] for i, record in enumerate(heldout)}
            from dataclasses import replace
            result = simulate(heldout, mapping, e, replace(io, prefetch_window_s=runtime['prefetch_window_s']), predictions=predictions)
            experiments.append(dict(page_size=size, graph_packed=graph, policy='task_profile_prefetch', **result))
    return {'atlas': atlas, 'experiments': experiments, 'fit_tokens': len(fit), 'evaluation_tokens': len(heldout),
            'io': io.__dict__, 'storage_measurement': 'Explicit measured values' if ssd_mib_s is not None else 'Configured assumptions; not measured',
            'scope': 'Micro routing, fixed bytes cache comparison, no model output changes. Large-model projection is a separate assumption-driven analysis.'}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/sparse_v3.yaml'); p.add_argument('--checkpoint', type=Path)
    p.add_argument('--trace', type=Path, default=Path('results/routes.jsonl')); p.add_argument('--tokens', type=int, default=4096)
    p.add_argument('--ssd-mib-s', type=float); p.add_argument('--ssd-latency-ms', type=float)
    a = p.parse_args(); cfg = load_config(a.config)
    if a.checkpoint: collect(cfg, a.checkpoint, a.trace, a.tokens)
    r = analyze(cfg, a.trace, a.ssd_mib_s, a.ssd_latency_ms); Path('results/cache_simulation.json').write_text(json.dumps(r, indent=2)); print(json.dumps({k:v for k,v in r.items() if k != 'atlas'}, indent=2))

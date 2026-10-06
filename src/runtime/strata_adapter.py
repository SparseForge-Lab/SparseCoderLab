"""Read-only native Strata trace parser. Does not import or modify Strata."""
from __future__ import annotations
import json, math, struct
from pathlib import Path

def parse_binary_trace(path: Path, layers: int = 48, experts: int = 256, max_records: int = 1000000):
    """Conservative format: layer-major dumps lack token IDs; never invent alignment.

    Returns dispatch records, not asserted token rows. A separate conversion requires
    exact regular layer cycles (single-token one-shot path, no speculative windows).
    """
    with path.open('rb') as f:
        for index in range(max_records):
            header = f.read(8)
            if not header: break
            if len(header) != 8: raise ValueError('Truncated trace header')
            layer, k = struct.unpack('<ii', header)
            if not 0 <= layer < layers or not 1 <= k <= 64: raise ValueError('Invalid layer/top-k')
            payload = f.read(k * 8)
            if len(payload) != k * 8: raise ValueError('Truncated trace payload')
            ids = list(struct.unpack_from(f'<{k}i', payload)); weights = list(struct.unpack_from(f'<{k}f', payload, k*4))
            if any(not 0 <= e < experts for e in ids) or any(not math.isfinite(w) for w in weights): raise ValueError('Invalid expert IDs/weights')
            yield {'dispatch': index, 'layer': layer, 'experts': ids, 'weights': weights}

def token_records(dispatches: list[dict], layers: int) -> list[dict]:
    if len(dispatches) % layers: raise ValueError('Incomplete layer cycles; cannot reconstruct token positions')
    result = []
    for start in range(0, len(dispatches), layers):
        group = dispatches[start:start + layers]
        if [r['layer'] for r in group] != list(range(layers)):
            raise ValueError('Non-single-token ordering (batched/speculative trace); token alignment unknown')
        result.append({'token': start // layers, 'layers': {str(r['layer']): r['experts'] for r in group},
                       'selected_weights': {str(r['layer']): r['weights'] for r in group},
                       'kind': 'external_strata', 'language': 'unknown', 'source': 'native --dump-routing, strict single-token cycles'})
    return result

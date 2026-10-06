from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    parent = cfg.pop('extends', None)
    if parent:
        base = load_config(path.parent / parent)
        def merge(a: dict, b: dict) -> dict:
            out = dict(a)
            for k, v in b.items(): out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
            return out
        cfg = merge(base, cfg)
    m = cfg['model']
    assert m['d_model'] == m['q_heads'] * m['head_dim']
    assert m['q_heads'] % m['kv_heads'] == 0 and m['head_dim'] % 2 == 0
    assert m['top_k'] in (1, 2) and m['top_k'] <= m['experts']
    assert all(0 <= i < m['layers'] for i in m['moe_layers'])
    assert cfg['training']['context'] <= cfg['training']['max_context']
    return cfg

def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

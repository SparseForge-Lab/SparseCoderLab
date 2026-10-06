from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

def write_trace(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True); count = 0
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            validate_record(record); f.write(json.dumps(record, sort_keys=True) + '\n'); count += 1
    return count

def validate_record(record: dict) -> None:
    if not isinstance(record.get('token'), int) or not isinstance(record.get('layers'), dict): raise ValueError('Invalid trace schema')
    for layer, experts in record['layers'].items():
        if not str(layer).isdigit() or not isinstance(experts, list) or not all(isinstance(e, int) and e >= 0 for e in experts):
            raise ValueError('Invalid expert IDs')

def read_trace(path: Path, max_records: int = 1000000):
    with path.open('r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            if index >= max_records: break
            record = json.loads(line); validate_record(record); yield record

def global_experts(record: dict, experts_per_layer: int) -> list[int]:
    return [int(layer) * experts_per_layer + e for layer, experts in sorted(record['layers'].items(), key=lambda x: int(x[0])) for e in experts]

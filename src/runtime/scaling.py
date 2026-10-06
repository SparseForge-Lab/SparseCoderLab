from __future__ import annotations
from src.moe.trace import global_experts

def scale_trace(records: list[dict], source_experts: int, capacity_layers: int, target_experts: int, rotate_tokens: int = 1024) -> list[dict]:
    """Synthetic topology sensitivity sweep, explicitly NOT measured large-model routing."""
    if min(source_experts,capacity_layers,target_experts,rotate_tokens)<1: raise ValueError('Invalid scale')
    rows=[]
    for index,record in enumerate(records):
        keys=sorted(record['layers'],key=int)
        if not keys: raise ValueError('Empty source routing trace')
        layers={}
        for layer in range(capacity_layers):
            source=record['layers'][keys[layer%len(keys)]]
            layers[str(layer)]=[(e+(index//rotate_tokens)*source_experts)%target_experts for e in source]
        rows.append({'token':index,'layers':layers,'kind':record.get('kind','unknown'),'language':record.get('language','unknown'),
                     'source':'Synthetic layer replication/rotating expert remap. Locality assumption, not measured target trace.'})
    return rows

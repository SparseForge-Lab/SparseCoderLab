"""Order-independent bounded document and position selection."""
import hashlib
import heapq
import json
from collections import defaultdict


def sample_documents(rows, limit, context, seed, stratum, minimum_length=2):
    if limit < 1 or context < 1: raise ValueError('Invalid evaluation sample size')
    heaps=defaultdict(list)
    for row in rows:
        if row['length'] < minimum_length: continue
        identity=json.dumps({k:row.get(k) for k in ('sha256','source_content_sha256','repository','file_path','start')},sort_keys=True)
        bits=hashlib.sha256(f'{seed}\0{identity}'.encode()).digest()
        rank=int.from_bytes(bits[:16],'big')
        key=stratum(row); selected=dict(row)
        selected['evaluation_offset']=int.from_bytes(bits[16:],'big') % (max(row['length']-context-1,0)+1)
        # The textual identity resolves rank ties without comparing dictionaries.
        entry=(-rank,identity,selected)
        if any(identity==old[1] for old in heaps[key]):
            continue
        if len(heaps[key]) < limit: heapq.heappush(heaps[key],entry)
        elif rank < -heaps[key][0][0]: heapq.heapreplace(heaps[key],entry)
    return {key:[row for _,_,row in sorted(heap,reverse=True)] for key,heap in sorted(heaps.items())}

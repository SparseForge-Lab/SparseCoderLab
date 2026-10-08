"""Convert packed split files into bounded physical shards without changing token order."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from src.config import load_config
from src.training.shards import TokenShardWriter
import numpy as np

def shard(cfg: dict) -> dict:
    directory=Path(cfg['data']['shards']); manifest_path=directory/'manifest.json'; manifest=json.loads(manifest_path.read_text())
    if manifest.get('physical_shards'): return manifest
    physical={}; shard_tokens=cfg['data']['shard_tokens']
    if shard_tokens<=0: raise ValueError('Invalid shard size')
    for split in ('train','val'):
        source=directory/f'{split}.bin'; writer=TokenShardWriter(directory,split,shard_tokens)
        if source.stat().st_size%2:raise ValueError('Partial uint16 token in source')
        with source.open('rb') as f:
            while data:=f.read(1024**2):writer.add(np.frombuffer(data,dtype=np.uint16))
        physical[split]=writer.close()
    # New manifests are written before deleting only the project-local generated monoliths.
    manifest['physical_shards']=physical; manifest_path.write_text(json.dumps(manifest,indent=2))
    Path(cfg['data']['manifest']).write_text(json.dumps(manifest,indent=2))
    root=Path.cwd().resolve()
    for split in ('train','val'):
        source=(directory/f'{split}.bin').resolve()
        if not source.is_relative_to(root): raise ValueError('Refusing to delete file outside project')
        source.unlink()
    return manifest

def verify_shards(cfg: dict) -> None:
    directory=Path(cfg['data']['shards']); manifest=json.loads((directory/'manifest.json').read_text())
    for split in ('train','val'):
        files=manifest.get('physical_shards',{}).get(split,[{'path':f'{split}.bin','sha256':manifest['shard_sha256'][split]}])
        for entry in files:
            h=hashlib.sha256()
            with (directory/entry['path']).open('rb') as f:
                while chunk:=f.read(1024**2): h.update(chunk)
            if h.hexdigest()!=entry['sha256']: raise ValueError('Shard checksum mismatch; resume would change data')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/dense_compute.yaml'); a=p.parse_args(); cfg=load_config(a.config)
    r=shard(cfg); verify_shards(cfg); print(json.dumps(r['physical_shards'],indent=2))

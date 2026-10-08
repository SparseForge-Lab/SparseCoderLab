"""Stream uint16 tokens directly to physical shards with bounded buffers."""
import hashlib
from pathlib import Path
import numpy as np

DEFAULT_SHARD_TOKENS = 134_217_728  # 256 MiB, eight shards per billion tokens.


class TokenShardWriter:
    def __init__(self, root, split, shard_tokens=DEFAULT_SHARD_TOKENS):
        if shard_tokens < 1: raise ValueError('Shard size must be positive')
        self.root=Path(root); self.split=split; self.shard_tokens=shard_tokens
        self.total=0; self.current=0; self.handle=None; self.files=[]; self.digest=None
    def _finish(self):
        if self.handle is None: return
        self.handle.close(); self.temporary.replace(self.path)
        self.files.append(dict(path=self.path.name,tokens=self.current,sha256=self.digest.hexdigest()))
        self.handle=None
    def add(self,tokens):
        data=np.asarray(tokens,dtype=np.uint16)
        while len(data):
            if self.handle is None:
                self.path=self.root/f'{self.split}-{len(self.files):05d}.bin'
                if self.path.exists(): raise FileExistsError('Refusing to overwrite a physical shard')
                self.temporary=self.path.with_suffix('.bin.part')
                self.handle=self.temporary.open('xb'); self.current=0; self.digest=hashlib.sha256()
            size=min(len(data),self.shard_tokens-self.current,524288)
            raw=data[:size].tobytes(); self.handle.write(raw); self.digest.update(raw)
            self.current+=size; self.total+=size; data=data[size:]
            if self.current==self.shard_tokens:self._finish()
    def close(self):
        self._finish(); return self.files

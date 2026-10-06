from __future__ import annotations
import hashlib, json, re
from pathlib import Path

class Archive:
    def __init__(self, directory: Path): self.directory = directory; directory.mkdir(parents=True, exist_ok=True)
    def put(self, block_id: int, text: str) -> str:
        if block_id < 0: raise ValueError('Negative block ID')
        pointer = f'<mem:{block_id:06d}>'; path = self.directory / f'{block_id:06d}.json'
        payload = {'id': block_id, 'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != payload: raise ValueError('Archive IDs immutable')
        path.write_text(json.dumps(payload), encoding='utf-8'); return pointer
    def get(self, pointer: str) -> str:
        match = re.fullmatch(r'<mem:(\d+)>', pointer)
        if not match: raise ValueError('Malformed memory pointer')
        payload = json.loads((self.directory / f'{int(match[1]):06d}.json').read_text(encoding='utf-8'))
        if hashlib.sha256(payload['text'].encode()).hexdigest() != payload['sha256']: raise ValueError('Archive corrupted')
        return payload['text']

def compact_fields(fields: dict[str, str], pointers: list[str], ratio: int) -> dict:
    if ratio < 1: raise ValueError('Invalid compaction ratio')
    keys = list(fields); keep = max(len(keys) // ratio, 1)
    return {'goal': 'recover exact task facts', 'fields': {k: fields[k] for k in keys[:keep]}, 'archive': list(pointers)}

class TeacherInterface:
    """Optional later SFT target generator; no API calls in V1."""
    enabled = False
    def generate(self, text: str) -> str: raise RuntimeError('Teacher disabled; configure a reviewed provider explicitly')

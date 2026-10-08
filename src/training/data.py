from __future__ import annotations
import hashlib, json, math, random
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

SPECIAL_TOKENS = ['<bos>', '<eos>', '<doc>', '<repo>', '<file>', '<tool_call>', '<tool_result>', '<compact>', '<mem>', '<retrieved>']
LANGUAGE_SAMPLES = {
 'Python': 'def total(values: list[int]) -> int:\n    return sum(values)\n',
 'C': '#include <stdio.h>\nint main(void) { printf("hello\\n"); return 0; }\n',
 'C++': '#include <vector>\nstd::vector<int> values{1, 2, 3};\n',
 'Rust': 'fn main() { let values: Vec<u32> = vec![1, 2, 3]; println!("{:?}", values); }\n',
 'Java': 'class Main { public static void main(String[] args) { System.out.println("hi"); } }\n',
 'Go': 'package main\nimport "fmt"\nfunc main() { fmt.Println("hello") }\n',
 'JavaScript': 'const sum = (xs) => xs.reduce((a, b) => a + b, 0);\n',
 'TypeScript': 'function count(xs: number[]): number { return xs.length; }\n',
 'HTML/CSS': '<html><style>.panel { color: #123abc; padding: 2px; }</style><div>Hello</div></html>\n',
 'SQL': 'SELECT name, COUNT(*) FROM users WHERE active = TRUE GROUP BY name;\n',
 'Bash': '#!/bin/bash\nfor file in *.py; do printf "%s\\n" "$file"; done\n',
 'JSON/YAML': '{"key": [1, 2, 3], "enabled": true}\nconfig:\n  timeout: 12\n',
 'English': 'A reliable experiment holds the data order fixed and measures uncertainty across seeds.\n',
}

def digest(text: str) -> str: return hashlib.sha256(text.encode('utf-8')).hexdigest()

def split_document(text: str, seed: int, eval_fraction: float, group: str | None = None) -> str:
    # Repository groups stay wholly within one split; legacy corpora retain the
    # content-hash policy so old experiment manifests remain reproducible.
    key = f'repository\0{group}' if group is not None else text
    value = int(digest(str(seed) + '\0' + key)[:16], 16) / 2**64
    return 'val' if value < eval_fraction else 'train'

def development_documents(cfg: dict):
    """Original CC0 synthetic fixtures; only engineering validation, no quality claims."""
    rng = random.Random(cfg['seed']); kinds = list(cfg['mixture']); weights = list(cfg['mixture'].values())
    languages = [k for k in LANGUAGE_SAMPLES if k != 'English']
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
    for i in range(cfg['dev_documents']):
        kind = rng.choices(kinds, weights=weights)[0]
        symbols = [''.join(rng.choices(alphabet, k=16)) for _ in range(12)]
        lang = rng.choice(languages) if kind == 'code' else 'English'
        if kind == 'code':
            text = LANGUAGE_SAMPLES[lang] + '\n' + '\n'.join(f'// symbol_{s} = {rng.randrange(100000)}' for s in symbols)
        elif kind == 'general':
            text = ('An experiment must distinguish correlation from causation. Measure uncertainty and record the conditions. '
                    'Evidence comes from controlled observations. Independent measurements can falsify a hypothesis.\n') + ' '.join(symbols)
        elif kind == 'technical':
            text = f'For a matrix A with {i % 97 + 1} rows, y = A x. A stable solver checks residuals and conditions.\n' + ' '.join(symbols)
        else:
            text = (f'<tool_result>error: function_{symbols[0]} undefined in file_{i}.py\n'
                    f'<compact>goal repair; constraint timeout={i % 31 + 1}; failed approach rename; '
                    f'exact symbol=function_{symbols[0]}; archive <mem:{i:06d}>\n') + ' '.join(symbols)
        yield {'id': f'dev-{i}', 'text': text, 'kind': kind, 'language': lang, 'license': 'CC0-1.0', 'source': 'original synthetic fixture'}

def bounded_jsonl(path: Path, max_bytes: int):
    used = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            used += len(line.encode('utf-8'))
            if used > max_bytes: raise RuntimeError('Input exceeds scan budget; refusing biased prefix ingestion')
            document = json.loads(line)
            if not all(k in document for k in ('text', 'license', 'source', 'kind', 'language')):
                raise ValueError('Every document requires explicit provenance, license, category and language')
            if document['license'] not in ('MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 'CC0-1.0', 'CC-BY-4.0', 'ISC'):
                raise ValueError('License requires explicit review; source rejected')
            yield document

class ShardedTokens:
    def __init__(self, files: list[Path], max_open: int = 8):
        if max_open < 1:
            raise ValueError('At least one shard cache slot required')
        self.files = list(files); self.max_open = max_open; self._cache = OrderedDict()
        sizes = [path.stat().st_size for path in self.files]
        if any(size % 2 for size in sizes):
            raise ValueError('Token shard must contain whole uint16 values')
        self.ends = np.cumsum([size // 2 for size in sizes], dtype=np.int64)
        self.length = int(self.ends[-1]) if len(self.ends) else 0
    def _array(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        array = np.memmap(self.files[index], dtype=np.uint16, mode='r')
        self._cache[index] = array
        if len(self._cache) > self.max_open:
            _, old = self._cache.popitem(last=False)
            old._mmap.close()
        return array
    def close(self):
        for array in self._cache.values():
            array._mmap.close()
        self._cache.clear()
    def __len__(self): return self.length
    def __getitem__(self, item: slice) -> np.ndarray:
        if not isinstance(item,slice) or item.step not in (None,1): raise ValueError('Contiguous slice required')
        start, stop, _ = item.indices(self.length)
        result = np.empty(max(stop - start, 0), dtype=np.uint16)
        offset = 0; index = bisect_right(self.ends, start)
        while start < stop:
            previous = int(self.ends[index - 1]) if index else 0
            end = min(stop, int(self.ends[index]))
            array = self._array(index)
            size = end - start
            # Copy before another lookup can evict this mmap. No escaping views.
            result[offset:offset + size] = array[start - previous:end - previous]
            start = end; offset += size; index = bisect_right(self.ends, start)
        return result

class PackedStream:
    """Checkpointable deterministic epoch shuffle over fixed packed-sequence offsets."""
    def __init__(self, directory: Path, split: str, context: int, seed: int, cursor: int = 0):
        self.context = context; self.seed = seed; self.cursor = cursor
        self.last_segment_ids = None
        self.last_prediction_count = 0
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf8')) if manifest_path.exists() else {}
        self.document_ends = None
        if manifest.get('packing_policy') == 'document_isolated_v1':
            starts, ends = [], []
            with (directory / f'{split}_documents.jsonl').open(encoding='utf8') as handle:
                for line in handle:
                    doc = json.loads(line); starts.append(doc['start']); ends.append(doc['start'] + doc['length'])
            if starts != [0] + ends[:-1] or not ends:
                raise ValueError('Document index must partition the token stream')
            self.document_ends = np.asarray(ends, dtype=np.int64)
        files=sorted(directory.glob(f'{split}-*.bin'))
        self.tokens=ShardedTokens(files) if files else np.memmap(directory/f'{split}.bin',dtype=np.uint16,mode='r')
        self.count = (len(self.tokens) - 1) // context
        if self.document_ends is not None and self.document_ends[-1] != len(self.tokens):
            raise ValueError('Document index length mismatch')
        if self.count < 1: raise ValueError(f'{split} has no full sequence at context={context}')
        self.epoch = -1; self.order = np.array([], dtype=np.int64)
    def next(self, batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []; segments = []
        for _ in range(batch):
            epoch, pos = divmod(self.cursor, self.count)
            if self.epoch != epoch:
                self.order = np.random.default_rng(self.seed + epoch).permutation(self.count); self.epoch = epoch
            start = int(self.order[pos]) * self.context
            rows.append(np.array(self.tokens[start:start + self.context + 1], dtype=np.int64)); self.cursor += 1
            if self.document_ends is not None:
                segments.append(np.searchsorted(self.document_ends, np.arange(start, start + self.context + 1), side='right'))
        data = torch.from_numpy(np.stack(rows)).to(device)
        targets = data[:, 1:]
        self.last_prediction_count = batch * self.context
        if segments:
            segment_array = np.stack(segments)
            self.last_prediction_count -= int(np.count_nonzero(segment_array[:, :-1] != segment_array[:, 1:]))
            ids = torch.from_numpy(segment_array).to(device)
            self.last_segment_ids = ids[:, :-1]
            targets = targets.masked_fill(ids[:, :-1] != ids[:, 1:], -100)
        return data[:, :-1], targets

def tokenizer_report(tokenizer: Tokenizer) -> dict:
    report = {}
    for language, sample in LANGUAGE_SAMPLES.items():
        encoded = tokenizer.encode(sample).ids
        assert tokenizer.decode(encoded, skip_special_tokens=False) == sample
        report[language] = {'characters': len(sample), 'tokens': len(encoded), 'tokens_per_character': len(encoded) / len(sample),
                            'tokens_per_line': len(encoded) / max(len(sample.splitlines()), 1)}
    return report

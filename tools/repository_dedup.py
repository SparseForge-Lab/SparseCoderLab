"""Repository-aware exact/MinHash deduplication before splitting or FIM.

LSH proposes candidates; exact shingle Jaccard confirms every removal. The
optional exhaustive audit is for small fixtures, not a corpus-scale guarantee.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import time
import tempfile
from collections import OrderedDict
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from src.utils.hashing import sha256_file

from tools.repository_data_smoke import validate_export
from tools.repository_fim import split_header

PRIME = (1 << 61) - 1
PERMUTATIONS = 64
BANDS = 16
ROWS = PERMUTATIONS // BANDS
LEXEMES = re.compile(r'\w+|[^\w\s]', re.UNICODE)
FORMAT = 'lexical5_minhash64_lsh16x4_v1'


def shingles(body: str) -> set[int]:
    tokens = LEXEMES.findall(body.replace('\r\n', '\n').replace('\r', '\n'))
    return {int.from_bytes(hashlib.blake2b('\0'.join(tokens[i:i + 5]).encode('utf-8'),
                                         digest_size=8).digest(), 'big') % PRIME
            for i in range(max(0, len(tokens) - 4))}


@lru_cache(maxsize=1)
def coefficients():
    pairs=[]
    for i in range(PERMUTATIONS):
        digest=hashlib.sha256(f'{FORMAT}\0{i}'.encode()).digest()
        pairs.append((1+int.from_bytes(digest[:8],'big')%(PRIME-1),int.from_bytes(digest[8:16],'big')%PRIME))
    return pairs


def signature(values: set[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError('MinHash requires nonempty shingles')
    # Exact 61-bit Mersenne reduction using bounded uint64 limbs. The result
    # matches Python big-integer arithmetic; no overflowing a*x vectorization.
    array=np.fromiter(values,dtype=np.uint64); low=array&((1<<31)-1); high=array>>31
    result=[]
    for a,b in coefficients():
        a0,a1=a&((1<<31)-1),a>>31
        cross=a0*high+a1*low
        total=a0*low+((cross&((1<<30)-1))<<31)+(cross>>30)+2*a1*high+b
        reduced=(total&PRIME)+(total>>61)
        reduced=np.where(reduced>=PRIME,reduced-PRIME,reduced)
        result.append(int(reduced.min()))
    return tuple(result)


class ShingleStore:
    """Private temporary disk spool; only eight candidate sets stay resident."""
    def __init__(self):
        self.directory=tempfile.TemporaryDirectory(prefix='sparsecoder-dedup-')
        self.lengths=[]; self.cache=OrderedDict()
    def __len__(self): return len(self.lengths)
    def add(self, values):
        index=len(self.lengths); self.lengths.append(len(values))
        np.fromiter(sorted(values),dtype=np.uint64).tofile(Path(self.directory.name)/f'{index}.bin')
    def __getitem__(self,index):
        if index not in self.cache:
            self.cache[index]=set(np.fromfile(Path(self.directory.name)/f'{index}.bin',dtype=np.uint64).tolist())
        self.cache.move_to_end(index)
        if len(self.cache)>8:self.cache.popitem(last=False)
        return self.cache[index]
    def close(self): self.directory.cleanup()


def band_keys(sig: tuple[int, ...]):
    for i in range(BANDS):
        yield i, sig[i * ROWS:(i + 1) * ROWS]


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    shared = len(a & b)
    return shared / (len(a) + len(b) - shared)


def family_ids(repositories: set[str], families: list[list[str]]) -> dict[str, str]:
    parents = {repo: repo for repo in repositories}

    def root(repo):
        while parents[repo] != repo:
            parents[repo] = parents[parents[repo]]
            repo = parents[repo]
        return repo

    for family in families:
        if len(family) < 2 or len(set(family)) != len(family):
            raise ValueError('Each declared fork family needs distinct repository IDs')
        if any(repo not in repositories for repo in family):
            raise ValueError('Fork-family map contains an unknown repository')
        for repo in family[1:]:
            a, b = sorted((root(family[0]), root(repo)))
            parents[b] = a
    return {repo: root(repo) for repo in sorted(repositories)}


def deduplicate(records: list[dict], *, threshold: float = .85, min_shingles: int = 64,
                families: list[list[str]] | None = None, audit_all_pairs: bool = False,
                progress=None):
    if not 0 < threshold <= 1 or min_shingles < 1:
        raise ValueError('Invalid similarity threshold or minimum shingle count')
    if audit_all_pairs and len(records) > 1000:
        raise ValueError('Exhaustive audit is limited to 1000 records')
    mapping = family_ids({record['repository'] for record in records}, families or [])
    retained, removed = [], []
    sets = ShingleStore()
    normalized_seen = {}
    short_seen = {}
    index = defaultdict(set)
    counts = Counter()
    ordered = sorted(records, key=lambda r: (r['repository'], r['revision'], r['file_path'], r['raw_sha256']))
    for processed, record in enumerate(ordered, 1):
        if progress is not None and (processed == 1 or processed % 1000 == 0):
            progress({'processed': processed, 'total': len(ordered), 'retained': len(retained), 'removed': len(removed)})
        if record.get('fim', {}).get('applied'):
            raise ValueError('Deduplication must run before FIM')
        _, body = split_header(record)
        normalized = body.replace('\r\n', '\n').replace('\r', '\n').rstrip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if record.get('normalized_sha256', digest) != digest:
            raise ValueError('Normalized source hash mismatch')
        duplicate = normalized_seen.get(digest)
        reason = 'exact_normalized'
        score = 1.0
        values = shingles(body)
        short_key = None
        if len(values) < min_shingles and record.get('language') == 'Python' and record.get('kind') == 'code':
            try:
                short_key = hashlib.sha256(ast.dump(ast.parse(body),include_attributes=False).encode()).hexdigest()
            except (SyntaxError, ValueError):
                pass
        if duplicate is None and short_key in short_seen:
            duplicate, score, reason = short_seen[short_key], None, 'short_ast_duplicate'
        sig = signature(values) if len(values) >= min_shingles else None
        if duplicate is None and sig is not None:
            candidates = set().union(*(index[key] for key in band_keys(sig)))
            counts['lsh_candidate_pairs'] += len(candidates)
            for candidate in sorted(candidates):
                previous = sets[candidate]
                if min(len(values), len(previous)) / max(len(values), len(previous)) < threshold:
                    continue
                counts['exact_jaccard_comparisons'] += 1
                similarity = jaccard(values, previous)
                if similarity >= threshold:
                    duplicate, score, reason = candidate, similarity, 'near_duplicate'
                    break
        if duplicate is not None:
            previous = retained[duplicate]
            removed.append({'repository': record['repository'], 'file_path': record['file_path'],
                            'raw_sha256': record['raw_sha256'], 'reason': reason, 'jaccard': score,
                            'retained_repository': previous['repository'],
                            'retained_file_path': previous['file_path'], 'retained_raw_sha256': previous['raw_sha256']})
            counts[reason] += 1
            counts['cross_repository_removals'] += int(record['repository'] != previous['repository'])
            continue
        out = copy.deepcopy(record)
        out['repository_family'] = mapping[record['repository']]
        out['deduplication'] = {'format': FORMAT, 'threshold': threshold, 'min_shingles': min_shingles,
                                'short_file_policy': 'Python AST identity v1'}
        position = len(retained)
        retained.append(out)
        sets.add(values)
        normalized_seen[digest] = position
        if short_key is not None: short_seen[short_key] = position
        if sig is not None:
            for key in band_keys(sig):
                index[key].add(position)
        else:
            counts['below_near_dedup_minimum'] += 1
    audit_pairs = 0
    if audit_all_pairs:
        for a in range(len(sets)):
            for b in range(a):
                if min(len(sets[a]), len(sets[b])) < min_shingles:
                    continue
                if min(len(sets[a]), len(sets[b])) / max(len(sets[a]), len(sets[b])) < threshold:
                    continue
                audit_pairs += 1
                if jaccard(sets[a], sets[b]) >= threshold:
                    raise ValueError('Exhaustive fixture audit found an LSH miss among retained records')
    counts.update({'exact_normalized': 0, 'near_duplicate': 0, 'short_ast_duplicate': 0})
    report = {'format': FORMAT, 'short_file_policy': 'Python AST identity v1; syntax-error and other-language short files retain exact-only policy',
              'shingle_storage': 'Temporary disk spool with eight-set LRU; input records and LSH index still in memory',
              'input_records': len(records), 'retained_records': len(retained),
              'counts': dict(counts), 'threshold': threshold, 'minimum_shingles': min_shingles,
              'permutations': PERMUTATIONS, 'bands': BANDS, 'rows_per_band': ROWS,
              'declared_fork_families': families or [], 'family_mapping': mapping,
              'exhaustive_retained_pair_audit': audit_all_pairs, 'audit_pairs_compared': audit_pairs,
              'estimated_lsh_candidate_probability_at_threshold': 1 - (1 - threshold ** ROWS) ** BANDS,
              'limitations': 'Lexical 5-shingle similarity, not semantic equivalence. Short Python removals require identical parsed AST; other short files receive exact dedup only. LSH may miss near pairs; exhaustive audit applies only when explicitly enabled. Fork families require reviewed declarations; file similarity is not proof of fork ancestry.'}
    sets.close()
    return retained, removed, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exports', nargs='+', type=Path, default=[
        Path('data/research_v2_real/exports/flask.jsonl'), Path('data/research_v2_real/exports/requests.jsonl')])
    parser.add_argument('--families', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/research_v2_real/documents/dedup_smoke.jsonl'))
    parser.add_argument('--removed', type=Path, default=Path('data/research_v2_real/documents/dedup_smoke_removed.jsonl'))
    parser.add_argument('--report', type=Path, default=Path('results/research_v2_real/dedup_smoke.json'))
    parser.add_argument('--threshold', type=float, default=.85)
    parser.add_argument('--audit-all-pairs', action='store_true')
    args = parser.parse_args()
    if any(path.exists() for path in (args.output, args.removed, args.report)):
        raise FileExistsError('Dedup fixture already exists; use versioned output paths')
    records, sources = [], []
    for export in sorted(args.exports):
        source, rows = validate_export(export)
        sources.append({'id': source['source_id'], 'revision': source['revision'],
                        'export_sha256': sha256_file(export)})
        records.extend(rows)
    families = json.loads(args.families.read_text(encoding='utf-8'))['families'] if args.families else []
    started = time.perf_counter()
    kept, removed, report = deduplicate(records, threshold=args.threshold, families=families,
                                      audit_all_pairs=args.audit_all_pairs)
    for path, rows in ((args.output, kept), (args.removed, removed)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    report.update(schema_version=1, passed=True, sources=sources,
                  preprocessing_seconds=time.perf_counter() - started,
                  output_sha256=sha256_file(args.output),
                  removed_inventory_sha256=sha256_file(args.removed),
                  script_sha256=sha256_file(Path(__file__)),
                  scope='Two-repository preprocessing fixture; full-corpus fork audit and benchmark contamination checks remain open.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'input': len(records), 'retained': len(kept),
                      'counts': report['counts'], 'audit_pairs': report['audit_pairs_compared']}, indent=2))


if __name__ == '__main__':
    main()

"""Repository-aware exact/MinHash deduplication before splitting or FIM.

LSH proposes candidates; exact shingle Jaccard confirms every removal. The
optional exhaustive audit is for small fixtures, not a corpus-scale guarantee.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

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


def signature(values: set[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError('MinHash requires nonempty shingles')
    result = []
    for i in range(PERMUTATIONS):
        digest = hashlib.sha256(f'{FORMAT}\0{i}'.encode()).digest()
        a = 1 + int.from_bytes(digest[:8], 'big') % (PRIME - 1)
        b = int.from_bytes(digest[8:16], 'big') % PRIME
        result.append(min((a * value + b) % PRIME for value in values))
    return tuple(result)


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
                families: list[list[str]] | None = None, audit_all_pairs: bool = False):
    if not 0 < threshold <= 1 or min_shingles < 1:
        raise ValueError('Invalid similarity threshold or minimum shingle count')
    if audit_all_pairs and len(records) > 1000:
        raise ValueError('Exhaustive audit is limited to 1000 records')
    mapping = family_ids({record['repository'] for record in records}, families or [])
    retained, removed = [], []
    sets = []
    normalized_seen = {}
    index = defaultdict(set)
    counts = Counter()
    ordered = sorted(records, key=lambda r: (r['repository'], r['revision'], r['file_path'], r['raw_sha256']))
    for record in ordered:
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
        out['deduplication'] = {'format': FORMAT, 'threshold': threshold, 'min_shingles': min_shingles}
        position = len(retained)
        retained.append(out)
        sets.append(values)
        normalized_seen[digest] = position
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
    report = {'format': FORMAT, 'input_records': len(records), 'retained_records': len(retained),
              'counts': dict(counts), 'threshold': threshold, 'minimum_shingles': min_shingles,
              'permutations': PERMUTATIONS, 'bands': BANDS, 'rows_per_band': ROWS,
              'declared_fork_families': families or [], 'family_mapping': mapping,
              'exhaustive_retained_pair_audit': audit_all_pairs, 'audit_pairs_compared': audit_pairs,
              'estimated_lsh_candidate_probability_at_threshold': 1 - (1 - threshold ** ROWS) ** BANDS,
              'limitations': 'Lexical 5-shingle similarity, not semantic equivalence. Small files receive exact dedup only. LSH may miss near pairs; exhaustive audit applies only when explicitly enabled. Fork families require reviewed declarations; file similarity is not proof of fork ancestry.'}
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
                        'export_sha256': hashlib.sha256(export.read_bytes()).hexdigest()})
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
                  output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  removed_inventory_sha256=hashlib.sha256(args.removed.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Two-repository preprocessing fixture; full-corpus fork audit and benchmark contamination checks remain open.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'input': len(records), 'retained': len(kept),
                      'counts': report['counts'], 'audit_pairs': report['audit_pairs_compared']}, indent=2))


if __name__ == '__main__':
    main()

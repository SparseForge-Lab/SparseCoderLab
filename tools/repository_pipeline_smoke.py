"""Validate import -> dedup -> FIM -> split -> pack -> stream cursor on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.config import load_config
from src.training.data import PackedStream
from tools.prepare_data import prepare
from tools.repository_data_smoke import validate_export
from tools.repository_dedup import deduplicate
from tools.repository_fim import MARKERS, restore, transform
from tools.shard_data import verify_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/research_v2_real/repo_fim_smoke.yaml')
    parser.add_argument('--exports', nargs='+', type=Path, default=[
        Path('data/research_v2_real/exports/flask.jsonl'), Path('data/research_v2_real/exports/requests.jsonl')])
    parser.add_argument('--report', type=Path, default=Path('results/research_v2_real/pipeline_fim_smoke.json'))
    parser.add_argument('--families', type=Path)
    parser.add_argument('--audit-all-pairs', action='store_true', help='Require exhaustive audit; limited to 1000 records')
    args = parser.parse_args()
    cfg = load_config(args.config)
    version = cfg['data']['dataset_version']
    output = Path('data/research_v2_real/documents') / (version + '.jsonl')
    if output.exists() or args.report.exists() or Path(cfg['data']['manifest']).exists():
        raise FileExistsError('Pipeline fixture already exists; use a new version and output paths')
    tokenizer_path = Path(cfg['data']['tokenizer'])
    before = tokenizer_path.read_bytes()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source_records, sources = [], []
    for export in sorted(args.exports):
        manifest, records = validate_export(export)
        source_records.extend(records)
        sources.append({'id': manifest['source_id'], 'revision': manifest['revision'],
                        'export_sha256': hashlib.sha256(export.read_bytes()).hexdigest()})
    families = json.loads(args.families.read_text(encoding='utf-8'))['families'] if args.families else []
    kept, removed, dedup = deduplicate(source_records, families=families,
                                      audit_all_pairs=args.audit_all_pairs or len(source_records) <= 1000)
    counts = Counter()
    language_stats = defaultdict(Counter)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        for original in kept:
            sample = transform(original, seed=cfg['data']['seed'], ratio=.4)
            if restore(sample) != original['text']:
                raise ValueError('FIM reconstruction mismatch')
            ids = tokenizer.encode(sample['text']).ids
            if tokenizer.decode(ids, skip_special_tokens=False) != sample['text']:
                raise ValueError('Frozen tokenizer roundtrip mismatch')
            counts['roundtrips'] += 1
            counts['eligible'] += int(sample['fim']['eligible'])
            counts['applied'] += int(sample['fim']['applied'])
            stats = language_stats[sample['language']]
            stats['documents'] += 1
            stats['tokens'] += len(ids)
            stats['characters'] += len(sample['text'])
            stats['utf8_bytes'] += len(sample['text'].encode('utf-8'))
            handle.write(json.dumps(sample, ensure_ascii=False) + '\n')
    manifest = prepare(cfg, output)
    verify_shards(cfg)
    by_split = defaultdict(Counter)
    families_by_split = defaultdict(set)
    repository_stats = defaultdict(Counter)
    role_stats = defaultdict(Counter)
    directory = Path(cfg['data']['shards'])
    for split in ('train', 'val'):
        with (directory / f'{split}_documents.jsonl').open(encoding='utf-8') as handle:
            lines = list(handle)
        for line in lines:
            record = json.loads(line)
            families_by_split[split].add(record['repository_family'])
            by_split[split]['documents'] += 1
            by_split[split]['eligible'] += int(record['fim']['eligible'])
            by_split[split]['applied'] += int(record['fim']['applied'])
            by_split[split]['tokens'] += record['length']
            repository_stats[record['repository']]['documents'] += 1
            repository_stats[record['repository']]['tokens'] += record['length']
            repository_stats[record['repository']]['fim'] += int(record['fim']['applied'])
            role_stats[record['role']]['documents'] += 1
            role_stats[record['role']]['tokens'] += record['length']
    if not families_by_split['train'] or not families_by_split['val']:
        raise ValueError('Fixture needs nonempty train and validation families')
    if families_by_split['train'] & families_by_split['val']:
        raise ValueError('Repository-family split contamination')
    cursor_receipts = {}
    for split in ('train', 'val'):
        stream = PackedStream(directory, split, context=1024, seed=cfg['data']['seed'])
        stream.next(2, 'cpu')
        saved_cursor = stream.cursor
        expected = stream.next(2, 'cpu')
        resumed = PackedStream(directory, split, context=1024, seed=cfg['data']['seed'], cursor=saved_cursor)
        actual = resumed.next(2, 'cpu')
        if not all(torch.equal(a, b) for a, b in zip(expected, actual)):
            raise ValueError('Packed stream cursor resume mismatch')
        cursor_receipts[split] = {'saved_cursor': saved_cursor, 'next_batch_exact': True,
                                  'context': 1024, 'batch': 2}
    if tokenizer_path.read_bytes() != before:
        raise ValueError('Tokenizer changed during preprocessing')
    for stats in language_stats.values():
        stats['tokens_per_character'] = stats['tokens'] / max(1, stats['characters'])
    receipt = {'schema_version': 1, 'passed': True, 'dataset_version': version,
               'sources': sources, 'dedup': dedup,
               'removed_source_hashes': [record['raw_sha256'] for record in removed],
               'fim': {'target_eligible_fraction': .4, 'counts': dict(counts),
                       'by_split': {key: dict(value) for key, value in by_split.items()},
                       'marker_token_ids': {marker: tokenizer.encode(marker).ids for marker in MARKERS},
                       'policy': 'Use existing frozen-tokenizer delimiters; no vocabulary additions.'},
               'repository_families_by_split': {key: sorted(value) for key, value in families_by_split.items()},
               'tokens': manifest['tokens'], 'stream_resume': cursor_receipts,
               'repository_packed_counts': {key: dict(value) for key, value in sorted(repository_stats.items())},
               'role_packed_counts': {key: dict(value) for key, value in sorted(role_stats.items())},
               'language_efficiency_after_dedup_and_fim': {key: dict(value) for key, value in sorted(language_stats.items())},
               'tokenizer_sha256': hashlib.sha256(before).hexdigest(),
               'manifest_sha256': hashlib.sha256(Path(cfg['data']['manifest']).read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'scope': 'CPU-only pinned-repository preprocessing/stream fixture. No model optimizer resume, checkpoint test, functional evaluation or training readiness certification.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'tokens': manifest['tokens'], 'fim': dict(counts),
                      'stream_resume_exact': True, 'vocabulary_additions': 0}, indent=2))


if __name__ == '__main__':
    main()

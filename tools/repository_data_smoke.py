"""Pack a small pinned-repository fixture without training or changing old data."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from src.utils.hashing import sha256_file

from tokenizers import Tokenizer

from src.config import load_config
from tools.prepare_data import prepare
from tools.shard_data import verify_shards


def sha256(path: Path) -> str:
    return sha256_file(path)


def validate_export(export: Path) -> tuple[dict, list[dict]]:
    source = json.loads(export.with_suffix(export.suffix + '.manifest.json').read_text(encoding='utf-8'))
    with export.open(encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle]
    if len(records) != source['included_files'] or export.stat().st_size != source['output_bytes']:
        raise ValueError('Export record count/size disagrees with its manifest')
    inventory = []
    for record in records:
        if record['repository'] != source['source_id'] or record['revision'] != source['revision']:
            raise ValueError('Export repository/revision disagrees with its manifest')
        prefix = f"<repo>{record['repository']}<file>{record['file_path']}\n"
        if not record['text'].startswith(prefix):
            raise ValueError('Export repository/file header mismatch')
        body = record['text'][len(prefix):]
        if hashlib.sha256(body.encode('utf-8')).hexdigest() != record['raw_sha256']:
            raise ValueError('Export raw-content hash mismatch')
        normalized = body.replace('\r\n', '\n').replace('\r', '\n').rstrip()
        if hashlib.sha256(normalized.encode('utf-8')).hexdigest() != record['normalized_sha256']:
            raise ValueError('Export normalized-content hash mismatch')
        inventory.append((record['file_path'], record['raw_sha256']))
    body = '\n'.join(f'{path}\0{digest}' for path, digest in sorted(inventory)).encode('utf-8')
    if hashlib.sha256(body).hexdigest() != source['file_inventory_sha256']:
        raise ValueError('Export file inventory disagrees with its manifest')
    return source, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/research_v2_real/repo_smoke.yaml')
    parser.add_argument('--exports', type=Path, nargs='+', default=[
        Path('data/research_v2_real/exports/flask.jsonl'),
        Path('data/research_v2_real/exports/requests.jsonl'),
    ])
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg['data'].get('dataset_version'):
        raise ValueError('Smoke corpus requires a version')
    if (Path(cfg['data']['shards']) / 'manifest.json').exists():
        raise FileExistsError('Smoke corpus is frozen; use a new version/config')
    destination = Path('data/research_v2_real/documents') / (cfg['data']['dataset_version'] + '.jsonl')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(cfg['data']['tokenizer'])
    before = sha256(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sources = []
    roles = collections.Counter()
    language_stats = collections.defaultdict(lambda: collections.Counter())
    roundtrips = 0
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        for export in sorted(args.exports):
            source, records = validate_export(export)
            sources.append({key: source[key] for key in (
                'source_id', 'source_url', 'revision', 'license_assertion', 'license_file_sha256',
                'included_files', 'output_bytes', 'filtered_counts', 'file_inventory_sha256', 'exporter_sha256',
            )} | {'export_sha256': sha256(export)})
            for record in records:
                encoded = tokenizer.encode(record['text']).ids
                if tokenizer.decode(encoded, skip_special_tokens=False) != record['text']:
                    raise ValueError('Frozen tokenizer failed an exact text roundtrip')
                roundtrips += 1
                roles[record['role']] += 1
                stats = language_stats[record['language']]
                stats['documents'] += 1
                stats['tokens'] += len(encoded)
                stats['characters'] += len(record['text'])
                stats['utf8_bytes'] += len(record['text'].encode('utf-8'))
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    manifest = prepare(cfg, destination)
    verify_shards(cfg)
    after = sha256(tokenizer_path)
    if before != after:
        raise ValueError('Frozen tokenizer changed during preprocessing')
    repositories = manifest['repositories_by_split']
    if not repositories.get('train') or not repositories.get('val'):
        raise ValueError('Fixture must exercise both train and validation')
    if set(repositories['train']) & set(repositories['val']):
        raise ValueError('Repository overlap between splits')
    for stats in language_stats.values():
        stats['tokens_per_character'] = stats['tokens'] / max(1, stats['characters'])
    receipt = {
        'schema_version': 1,
        'passed': True,
        'dataset_version': manifest['dataset_version'],
        'sources': sources,
        'tokenizer_sha256': before,
        'tokenizer_unchanged': True,
        'exact_tokenizer_roundtrips': roundtrips,
        'role_document_counts_before_dedup': dict(roles),
        'language_efficiency_before_dedup': {key: dict(value) for key, value in sorted(language_stats.items())},
        'tokens': manifest['tokens'],
        'document_stats': manifest['document_stats'],
        'rejected': manifest['rejected'],
        'repositories_by_split': repositories,
        'physical_shards_verified': True,
        'source_inventory_verified': True,
        'manifest_sha256': sha256(Path(cfg['data']['manifest'])),
        'scope': 'Two-repository preprocessing fixture only. No model training or functional coding evaluation. Does not certify the full corpus, near deduplication, FIM or a long-run readiness gate.',
    }
    output = Path('results/research_v2_real/repository_smoke.json')
    output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'output': output.as_posix(), 'tokens': receipt['tokens'],
                      'roundtrips': roundtrips, 'repositories_by_split': repositories}, indent=2))


if __name__ == '__main__':
    main()

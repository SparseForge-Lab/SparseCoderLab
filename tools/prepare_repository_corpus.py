"""Build a new version from reviewed pinned source subsets; never run corpus code."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from src.config import load_config
from src.training.data import PackedStream
from src.utils.hashing import sha256_file
from tools.import_local_repo import run_git
from tools.prepare_data import prepare
from tools.repository_data_smoke import validate_export
from tools.repository_dedup import deduplicate
from tools.repository_fim import restore, transform
from tools.shard_data import verify_shards


def progress(stage, **values):
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def verify_source(source, root):
    if run_git(root, 'rev-parse', 'HEAD') != source['revision']:
        raise ValueError(f"Pinned revision mismatch: {source['id']}")
    import hashlib
    for evidence in source['license_evidence']:
        raw = subprocess.check_output(['git', '-C', str(root), 'show', f"{source['revision']}:{evidence['path']}"])
        if hashlib.sha256(raw).hexdigest() != evidence['sha256']:
            raise ValueError(f"License evidence mismatch: {source['id']}")
    if source['github_fork'] and source['repository_family'] == source['id']:
        raise ValueError('Fork requires a reviewed shared family')


def export_sources(plan, paths, directory):
    """Reuse only exports with the same pin, license/filter policy and exporter."""
    directory.mkdir(parents=True, exist_ok=True)
    exports = []
    for source in plan['sources']:
        root = Path(paths[source['id']])
        verify_source(source, root)
        output = directory / (source['id'].replace('/', '__') + '.jsonl')
        additions = source.get('additional_extensions', {})
        if not output.exists():
            extension_map = directory / (output.stem + '.extensions.json')
            extension_map.write_text(json.dumps(additions), encoding='utf8')
            command = [sys.executable, '-m', 'tools.import_local_repo', '--repo', str(root),
                       '--source-id', source['id'], '--source-url', source['url'],
                       '--license', source['license_assertion'], '--license-file', source['license_file'],
                       '--extension-map', str(extension_map), '--output', str(output), '--max-mib', '1024']
            for prefix in source['excluded_prefixes']: command.extend(['--exclude-prefix', prefix])
            if source.get('conservative_license_review'): command.append('--conservative-license-review')
            if source['primary_language'] == 'C++': command.extend(['--header-language', 'C++'])
            subprocess.run(command, check=True, capture_output=True)
        receipt = json.loads(output.with_suffix('.jsonl.manifest.json').read_text(encoding='utf8'))
        expected = dict(revision=source['revision'], license_assertion=source['license_assertion'],
                        license_file_sha256=source['root_license_sha256'],
                        reviewed_exclusion_prefixes=source['excluded_prefixes'], additional_extensions=additions,
                        conservative_license_review=source.get('conservative_license_review', False),
                        exporter_sha256=sha256_file(Path('tools/import_local_repo.py')))
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Export identity/policy changed; choose a new export version: {source['id']}")
        exports.append((source, output))
        progress('export', source=source['id'], files=receipt['included_files'], bytes=receipt['output_bytes'])
    return exports


def build(cfg, plan, exports, report, plan_path):
    version = cfg['data']['dataset_version']
    output = Path('data/research_v2_real/documents') / (version + '.jsonl')
    if any(p.exists() for p in (output, report, Path(cfg['data']['manifest']))):
        raise FileExistsError('Corpus already exists; choose new versioned outputs')
    before = sha256_file(Path(cfg['data']['tokenizer']))
    tokenizer = Tokenizer.from_file(cfg['data']['tokenizer'])
    records, sources = [], []
    for source, path in exports:
        manifest, rows = validate_export(path)
        if source['primary_language'] == 'reasoning':
            for row in rows:
                if row['language'] in ('LaTeX', 'Lean', 'reStructuredText', 'Markdown'):
                    row.update(kind='general', role='reasoning_exposition')
        records.extend(rows)
        sources.append(dict(id=source['id'], revision=source['revision'],
                            export_sha256=sha256_file(path), files=len(rows), filtered=manifest['filtered_counts']))
    families = defaultdict(list)
    for source in plan['sources']: families[source['repository_family']].append(source['id'])
    progress('dedup_start', documents=len(records))
    kept, removed, dedup = deduplicate(records, families=[v for v in families.values() if len(v) > 1],
        progress=lambda values: progress('dedup', **values))
    del records
    progress('dedup_complete', retained=len(kept), removed=len(removed))
    counts = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8', newline='\n') as handle:
        for index, original in enumerate(kept, 1):
            sample = transform(original, seed=cfg['data']['seed'], ratio=.4, tokenizer=tokenizer)
            if restore(sample) != original['text']: raise ValueError('FIM reconstruction mismatch')
            ids = tokenizer.encode(sample['text']).ids
            if tokenizer.decode(ids, skip_special_tokens=False) != sample['text']:
                raise ValueError('Frozen tokenizer roundtrip mismatch')
            counts['roundtrips'] += 1
            counts['eligible'] += int(sample['fim']['eligible'])
            counts['applied'] += int(sample['fim']['applied'])
            handle.write(json.dumps(sample, ensure_ascii=False) + '\n')
            if index % 1000 == 0: progress('fim', processed=index, total=len(kept))
    del kept
    progress('packing_start')
    manifest = prepare(cfg, output)
    verify_shards(cfg)
    by_split = defaultdict(Counter)
    families_by_split = defaultdict(set)
    languages = defaultdict(Counter)
    repository_stats = defaultdict(Counter)
    directory = Path(cfg['data']['shards'])
    for split in ('train', 'val'):
        with (directory / f'{split}_documents.jsonl').open(encoding='utf8') as handle:
            for line in handle:
                row = json.loads(line)
                families_by_split[split].add(row['repository_family'])
                by_split[split]['documents'] += 1
                by_split[split]['eligible'] += int(row['fim']['eligible'])
                by_split[split]['applied'] += int(row['fim']['applied'])
                languages[split][row['language']] += row['length']
                repository_stats[row['repository']]['tokens'] += row['length']
                repository_stats[row['repository']]['documents'] += 1
    if not all(families_by_split.values()) or families_by_split['train'] & families_by_split['val']:
        raise ValueError('Empty split or family overlap')
    cursor_receipts = {}
    for split in ('train', 'val'):
        stream = PackedStream(directory, split, context=1024, seed=cfg['data']['seed'])
        stream.next(2, 'cpu'); cursor = stream.cursor
        expected = stream.next(2, 'cpu')
        resumed = PackedStream(directory, split, context=1024, seed=cfg['data']['seed'], cursor=cursor)
        if not all(torch.equal(a, b) for a, b in zip(expected, resumed.next(2, 'cpu'))):
            raise ValueError('Stream cursor mismatch')
        cursor_receipts[split] = dict(cursor=cursor, next_batch_exact=True)
    if sha256_file(Path(cfg['data']['tokenizer'])) != before: raise ValueError('Tokenizer changed')
    receipt = dict(schema_version=1, preprocessing_passed=True, dataset_version=version,
        source_plan_sha256=sha256_file(plan_path), sources=sources, dedup=dedup,
        removed_records=removed, fim=dict(counts), fim_by_split={k:dict(v) for k,v in by_split.items()},
        tokens=manifest['tokens'], token_mixture=manifest['token_mixture'],
        language_tokens_by_split={k:dict(v) for k,v in languages.items()},
        repository_packed_counts={k:dict(v) for k,v in repository_stats.items()},
        repository_families_by_split={k:sorted(v) for k,v in families_by_split.items()},
        stream_resume=cursor_receipts, tokenizer_sha256=before,
        manifest_sha256=sha256_file(Path(cfg['data']['manifest'])), script_sha256=sha256_file(Path(__file__)),
        scope='Corpus preprocessing and CPU cursor evidence; training readiness requires separate contamination, storage, evaluation and CUDA transition gates.')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    progress('complete', tokens=manifest['tokens'], fim=dict(counts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True, help='Private source-id/local-repository map')
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if shutil.disk_usage(args.plan).free < 30*1024**3: raise RuntimeError('Free disk reserve below 30 GiB')
    started = time.perf_counter()
    plan = json.loads(args.plan.read_text(encoding='utf8'))
    exports = export_sources(plan, json.loads(args.paths.read_text(encoding='utf8')), args.exports)
    build(load_config(args.config), plan, exports, args.report, args.plan)
    progress('elapsed', seconds=time.perf_counter()-started)


if __name__ == '__main__': main()

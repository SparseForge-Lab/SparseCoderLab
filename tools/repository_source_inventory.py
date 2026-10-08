"""Publish source identities and aggregate coverage without source payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from src.utils.hashing import sha256_file

from tools.repository_data_smoke import validate_export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('data/research_v2_real/exports'))
    parser.add_argument('--output', type=Path, default=Path('results/research_v2_real/source_inventory_pilot.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Source inventory already exists; use a versioned output path')
    sources = []
    languages, roles, filters = Counter(), Counter(), Counter()
    for export in sorted(args.directory.glob('*.jsonl')):
        if export.name.endswith('.quarantine.jsonl'):
            continue
        manifest, records = validate_export(export)
        lang = Counter(record['language'] for record in records)
        role = Counter(record['role'] for record in records)
        languages.update(lang)
        roles.update(role)
        filters.update(manifest['filtered_counts'])
        sources.append({key: manifest[key] for key in (
            'source_id', 'source_url', 'revision', 'license_assertion', 'license_file_sha256',
            'included_files', 'output_bytes', 'filtered_counts', 'file_inventory_sha256', 'exporter_sha256')}
                       | {'license_file': manifest.get('license_file'), 'language_file_counts': dict(lang),
                          'role_file_counts': dict(role), 'export_sha256': sha256_file(export),
                          'status': 'pinned_preprocessing_pilot',
                          'licensing_review': 'Root declaration recorded; obvious conflicting SPDX/GNU headers filtered. No claim of a complete per-file exception audit.'})
    report = {'schema_version': 1, 'sources': sources, 'repositories': len(sources),
              'files_before_global_dedup': sum(source['included_files'] for source in sources),
              'export_bytes': sum(source['output_bytes'] for source in sources),
              'language_file_counts': dict(sorted(languages.items())), 'role_file_counts': dict(sorted(roles.items())),
              'filter_counts': dict(sorted(filters.items())),
              'source_payloads_public': False,
              'readiness': 'Inventory pilot only; mixture, full licensing/fork/contamination review and training preflight remain open.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'repositories': report['repositories'], 'files': report['files_before_global_dedup'],
                      'export_bytes': report['export_bytes'], 'languages': report['language_file_counts']}))


if __name__ == '__main__':
    main()

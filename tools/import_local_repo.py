"""Export a clean, explicitly licensed Git revision as provenance-rich JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

EXTENSIONS = {
    '.py': 'Python', '.c': 'C', '.h': 'C', '.cc': 'C++', '.cpp': 'C++', '.hpp': 'C++',
    '.rs': 'Rust', '.java': 'Java', '.go': 'Go', '.js': 'JavaScript', '.jsx': 'JavaScript',
    '.ts': 'TypeScript', '.tsx': 'TypeScript', '.html': 'HTML/CSS', '.css': 'HTML/CSS',
    '.sql': 'SQL', '.sh': 'Bash', '.bash': 'Bash', '.json': 'JSON/YAML', '.yaml': 'JSON/YAML',
    '.yml': 'JSON/YAML', '.cs': 'C#', '.toml': 'TOML', '.md': 'Markdown',
    '.rst': 'reStructuredText', '.txt': 'Text', '.cmake': 'CMake', '.bats': 'Bash',
}
SPECIAL_FILES = {'Makefile': 'Makefile', 'Dockerfile': 'Dockerfile', 'CMakeLists.txt': 'CMake'}
LICENSES = ('MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 'CC0-1.0', 'ISC')
SKIP_PARTS = {
    'vendor', 'vendors', 'node_modules', 'dist', 'build', 'target', 'third_party',
    'third-party', 'generated', '__generated__', '.venv', 'venv', '.codex', '.claude',
}
SECRET_PATTERNS = [
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    re.compile(rb'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    re.compile(rb'(?i)(?:api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*["\'][^"\']{12,}["\']'),
]


def run_git(root: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_role(name: str) -> tuple[str, str]:
    parts = {part.casefold() for part in Path(name).parts}
    base = Path(name).name.casefold()
    if Path(name).suffix.lower() in ('.md', '.rst', '.txt'):
        return 'technical', 'documentation'
    if parts & {'test', 'tests', '__tests__', 'testing'} or base.startswith('test_') or base.endswith(('_test.go', '.test.js', '.spec.ts')):
        return 'code', 'test'
    if Path(name).name in SPECIAL_FILES or base in {'package.json', 'pyproject.toml', 'cargo.toml', 'go.mod'}:
        return 'context', 'build'
    if Path(name).suffix.lower() in ('.json', '.yaml', '.yml', '.toml'):
        return 'context', 'configuration'
    return 'code', 'implementation'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--source-id', required=True, help='Stable identifier such as owner/repository')
    parser.add_argument('--source-url', required=True, help='Canonical HTTPS repository URL')
    parser.add_argument('--license', required=True, choices=LICENSES)
    parser.add_argument('--license-file', help='Explicit root-relative license file, e.g. LICENSE-MIT')
    parser.add_argument('--exclude-prefix', action='append', default=[], help='Reviewed root-relative path prefix to omit')
    parser.add_argument('--header-language', choices=('C', 'C++'), default='C')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-mib', type=int, default=64)
    args = parser.parse_args()

    root = args.repo.resolve()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.source_id):
        raise ValueError('--source-id must be a stable owner/repository identifier, not a local path')
    source_url = urlparse(args.source_url)
    if source_url.scheme != 'https' or not source_url.hostname or source_url.username or source_url.password or source_url.query or source_url.fragment:
        raise ValueError('--source-url must be a canonical HTTPS URL without credentials, query or fragment')
    if args.max_mib <= 0:
        raise ValueError('--max-mib must be positive')
    revision = run_git(root, 'rev-parse', 'HEAD')
    if run_git(root, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Repository has tracked modifications; export a clean committed revision')

    for prefix in args.exclude_prefix:
        if not prefix or Path(prefix).is_absolute() or '..' in Path(prefix).parts or '\\' in prefix:
            raise ValueError('Exclusion prefixes must be safe POSIX repository-relative paths')
    if args.license_file:
        candidate = Path(args.license_file)
        if candidate.is_absolute() or '..' in candidate.parts or len(candidate.parts) != 1:
            raise ValueError('License file must be a root-relative filename')
        license_path = root / candidate
        if not license_path.is_file():
            raise ValueError('Selected license file missing')
    else:
        license_path = next((root / name for name in ('LICENSE', 'LICENSE.txt', 'LICENSE.md', 'License.txt', 'COPYING') if (root / name).is_file()), None)
    if license_path is None:
        raise ValueError('Repository license file missing')
    try:
        run_git(root, 'ls-files', '--error-unmatch', license_path.relative_to(root).as_posix())
    except subprocess.CalledProcessError as exc:
        raise ValueError('Repository license file must be tracked in the pinned revision') from exc
    license_raw = subprocess.check_output(['git', '-C', str(root), 'show', f'{revision}:{license_path.name}'])
    license_sha = sha256(license_raw)
    # Read committed blobs, rather than checkout bytes or symlink targets.
    tree = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', '-l', '-z', revision])
    candidates = []
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        header, raw_name = entry.split(b'\t', 1)
        mode, kind, oid, size = header.split()
        name = raw_name.decode('utf-8')
        candidates.append((name, mode, kind, oid, size))
    candidates.sort(key=lambda entry: entry[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path = args.output.with_suffix(args.output.suffix + '.quarantine.jsonl')
    manifest_path = args.output.with_suffix(args.output.suffix + '.manifest.json')
    if any(path.exists() for path in (args.output, quarantine_path, manifest_path)):
        raise FileExistsError('Export already exists; choose a new version/output path')
    output_tmp = args.output.with_suffix(args.output.suffix + '.tmp')
    quarantine_tmp = quarantine_path.with_suffix(quarantine_path.suffix + '.tmp')
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + '.tmp')
    counts: Counter[str] = Counter()
    file_hashes: list[tuple[str, str]] = []
    written = 0
    total_bytes = 0
    max_bytes = args.max_mib * 1024**2

    with output_tmp.open('x', encoding='utf-8', newline='\n') as out, quarantine_tmp.open('x', encoding='utf-8', newline='\n') as quarantine:
        for index, (name, mode, blob_kind, oid, blob_size) in enumerate(candidates):
            parts = Path(name).parts
            if mode not in (b'100644', b'100755') or blob_kind != b'blob':
                counts['symlink_or_submodule'] += 1
                continue
            if any(part.casefold() in SKIP_PARTS for part in parts) or Path(name).name.casefold().endswith('.min.js'):
                counts['filtered_path'] += 1
                continue
            if Path(name).name.casefold() in {'agents.md', 'claude.md'} or any(name.startswith(prefix) for prefix in args.exclude_prefix):
                counts['reviewed_exclusion'] += 1
                continue
            language = SPECIAL_FILES.get(Path(name).name, EXTENSIONS.get(Path(name).suffix.lower()))
            if Path(name).suffix.lower() == '.h':
                language = args.header_language
            if language is None:
                counts['unsupported_extension'] += 1
                continue
            if int(blob_size) > 1024**2:
                counts['oversize_or_missing'] += 1
                continue
            raw = subprocess.check_output(['git', '-C', str(root), 'cat-file', 'blob', oid.decode('ascii')])
            if any(pattern.search(raw) for pattern in SECRET_PATTERNS):
                quarantine.write(json.dumps({'path': name, 'reason': 'secret_pattern'}, ensure_ascii=False) + '\n')
                counts['secret_quarantined'] += 1
                continue
            try:
                text = raw.decode('utf-8')
            except UnicodeError:
                counts['invalid_utf8'] += 1
                continue
            if '\x00' in text:
                counts['binary'] += 1
                continue
            if re.search(r'auto.?generated|do not edit|generated by', text[:1000], re.I):
                counts['generated_header'] += 1
                continue
            if re.search(r'GNU (?:GENERAL|LESSER|AFFERO) PUBLIC LICENSE', text[:3000], re.I):
                counts['conflicting_license_header'] += 1
                continue
            spdx = re.search(r'SPDX-License-Identifier:\s*([^\r\n]+)', text[:3000])
            expression = spdx.group(1).strip().rstrip('*/ ').strip() if spdx else None
            file_license = args.license
            if expression:
                if expression in LICENSES:
                    file_license = expression
                elif re.fullmatch(r'[A-Za-z0-9.\- ()]+', expression) and ' OR ' in expression:
                    options = {piece.strip(' ()') for piece in expression.split(' OR ')}
                    if args.license not in options:
                        counts['unreviewed_spdx_expression'] += 1
                        continue
                else:
                    counts['unreviewed_spdx_expression'] += 1
                    continue
            kind, role = file_role(name)
            record = {
                'text': f'<repo>{args.source_id}<file>{name}\n{text}',
                'source': args.source_id,
                'source_url': args.source_url.rstrip('/'),
                'repository': args.source_id,
                'revision': revision,
                'file_path': name,
                'raw_sha256': sha256(raw),
                'normalized_sha256': sha256(text.replace('\r\n', '\n').replace('\r', '\n').rstrip().encode('utf-8')),
                'git_blob': oid.decode('ascii'),
                'source_license_sha256': license_sha,
                'license': file_license,
                'license_basis': 'spdx_header' if expression else 'repository_root_assertion',
                'license_expression': expression,
                'kind': kind,
                'role': role,
                'language': language,
            }
            line = json.dumps(record, ensure_ascii=False) + '\n'
            size = len(line.encode('utf-8'))
            if total_bytes + size > max_bytes:
                counts['budget_skipped'] += len(candidates) - index
                break
            out.write(line)
            written += 1
            total_bytes += size
            file_hashes.append((name, record['raw_sha256']))
            counts['included'] += 1

    inventory = '\n'.join(f'{name}\0{digest}' for name, digest in file_hashes).encode('utf-8')
    manifest = {
        'schema_version': 1,
        'source_id': args.source_id,
        'source_url': args.source_url.rstrip('/'),
        'revision': revision,
        'license_assertion': args.license,
        'license_file_sha256': license_sha,
        'license_file': license_path.name,
        'reviewed_exclusion_prefixes': args.exclude_prefix,
        'header_language': args.header_language,
        'included_files': written,
        'output_bytes': total_bytes,
        'filtered_counts': dict(sorted(counts.items())),
        'file_inventory_sha256': sha256(inventory),
        'exporter_sha256': sha256(Path(__file__).read_bytes()),
        'raw_content_policy': 'Bytes from the pinned Git commit; checkout line endings and symlinks are not read.',
        'max_output_bytes': max_bytes,
        'quarantine_metadata': quarantine_path.name,
        'license_note': 'Repository-level license is an explicit assertion; review per-file exceptions before training.',
    }
    with manifest_tmp.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(manifest, indent=2) + '\n')
    output_tmp.replace(args.output)
    quarantine_tmp.replace(quarantine_path)
    manifest_tmp.replace(manifest_path)
    print(json.dumps({'included_files': written, 'output_bytes': total_bytes, 'revision': revision,
                      'manifest': args.output.with_suffix(args.output.suffix + '.manifest.json').name,
                      'filtered_counts': dict(sorted(counts.items()))}, indent=2))


if __name__ == '__main__':
    main()

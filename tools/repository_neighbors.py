"""Conservative local dependency links and bounded multi-file corpus fixtures."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import posixpath
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from tokenizers import Tokenizer

from tools.repository_fim import restore, split_header


def python_modules(paths):
    result = defaultdict(set)
    for path in paths:
        if not path.endswith('.py'):
            continue
        components = list(PurePosixPath(path).with_suffix('').parts)
        if components[0] == 'src':
            components = components[1:]
        if components[-1] == '__init__':
            components = components[:-1]
        if components:
            result['.'.join(components)].add(path)
    return result


def graph(records):
    grouped = defaultdict(dict)
    for record in records:
        key = record['repository'], record['revision']
        path = record['file_path']
        if path in grouped[key]:
            raise ValueError('Duplicate file identity in dependency inventory')
        grouped[key][path] = record
    edges, diagnostics = [], Counter()
    for (repo, revision), files in sorted(grouped.items()):
        modules = python_modules(files)
        for path, record in sorted(files.items()):
            original = dict(record, text=restore(record))
            _, body = split_header(original)
            matches = set()
            language = record['language']
            if language == 'Python':
                try:
                    tree = ast.parse(body)
                except (SyntaxError, ValueError):
                    diagnostics['python_parse_skipped'] += 1
                    continue
                components = list(PurePosixPath(path).with_suffix('').parts)
                if components[0] == 'src':
                    components = components[1:]
                package = components[:-1]
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        if node.level:
                            if node.level > len(package):
                                diagnostics['relative_import_outside_package'] += 1
                                continue
                            prefix = package[:len(package) - node.level + 1]
                            module = '.'.join(prefix + ([node.module] if node.module else []))
                        else:
                            module = node.module or ''
                        names = [module] + [module + '.' + alias.name for alias in node.names if alias.name != '*']
                    for name in names:
                        candidates = modules.get(name, set())
                        if len(candidates) == 1:
                            matches.add((next(iter(candidates)), 'python_import'))
                        elif len(candidates) > 1:
                            diagnostics['ambiguous_module_skipped'] += 1
            elif language in {'JavaScript', 'TypeScript'}:
                imports = re.findall(r'''(?:from\s*|require\s*\(\s*|import\s*)["'](\.[^"']+)["']''', body)
                for imported in imports:
                    base = posixpath.normpath(posixpath.join(posixpath.dirname(path), imported))
                    if base.startswith('../') or base.startswith('/'):
                        continue
                    candidates = [base] + [base + suffix for suffix in ('.ts', '.tsx', '.js', '.jsx', '/index.ts', '/index.js')]
                    resolved = [candidate for candidate in candidates if candidate in files]
                    if len(resolved) == 1:
                        matches.add((resolved[0], 'relative_js_ts_import'))
            elif language in {'C', 'C++'}:
                for imported in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', body, re.M):
                    candidates = {posixpath.normpath(posixpath.join(posixpath.dirname(path), imported)),
                                  imported, 'include/' + imported}
                    resolved = sorted(candidate for candidate in candidates if candidate in files)
                    if len(resolved) == 1:
                        matches.add((resolved[0], 'quoted_c_cpp_include'))
            elif language == 'Rust':
                for name in re.findall(r'^\s*(?:pub(?:\([^\n]*\))?\s+)?mod\s+(\w+)\s*;', body, re.M):
                    base = posixpath.dirname(path)
                    if PurePosixPath(path).name not in {'lib.rs', 'main.rs', 'mod.rs'}:
                        base = posixpath.join(base, PurePosixPath(path).stem)
                    candidates = [posixpath.join(base, name + '.rs'), posixpath.join(base, name, 'mod.rs')]
                    resolved = [candidate for candidate in candidates if candidate in files]
                    if len(resolved) == 1:
                        matches.add((resolved[0], 'rust_sibling_module_candidate'))
            else:
                diagnostics['unsupported_language_files'] += 1
            for target, method in sorted(matches):
                if target == path:
                    continue
                edges.append({'repository': repo, 'revision': revision, 'source_path': path,
                              'target_path': target, 'source_sha256': record['raw_sha256'],
                              'target_sha256': files[target]['raw_sha256'], 'method': method,
                              'source_role': record['role'], 'target_role': files[target]['role']})
    return edges, dict(diagnostics)


def samples(records, edges, tokenizer, *, max_tokens=8192, max_neighbors=3):
    if max_tokens < 1 or max_neighbors < 1:
        raise ValueError('Sample token and neighbor limits must be positive')
    files = {(r['repository'], r['revision'], r['file_path']): r for r in records}
    neighbors = defaultdict(list)
    for edge in edges:
        key = edge['repository'], edge['revision'], edge['source_path']
        neighbors[key].append(edge['target_path'])
    for key, targets in sorted(neighbors.items()):
        anchor = files[key]
        selected = [anchor]
        text = restore(anchor)
        if len(tokenizer.encode(text).ids) > max_tokens:
            continue
        for target in sorted(set(targets))[:max_neighbors]:
            neighbor = files[key[:2] + (target,)]
            candidate = text + '\n<doc>\n' + restore(neighbor)
            if len(tokenizer.encode(candidate).ids) <= max_tokens:
                selected.append(neighbor)
                text = candidate
        if len(selected) < 2:
            continue
        yield {'text': text, 'kind': 'context', 'role': 'dependency_context', 'language': anchor['language'],
               'source': anchor['source'], 'source_url': anchor['source_url'],
               'repository': key[0], 'revision': key[1], 'file_path': key[2],
               'repository_family': anchor.get('repository_family', key[0]),
               'license': anchor['license'], 'sample_sha256': hashlib.sha256(text.encode()).hexdigest(),
               'license_set': sorted({r['license'] for r in selected}),
               'member_files': [{'path': r['file_path'], 'sha256': r['raw_sha256'], 'license': r['license'],
                                 'source_license_sha256': r['source_license_sha256']} for r in selected],
               'token_count': len(tokenizer.encode(text).ids),
               'format': 'repository_dependency_context_v1',
               'scope': 'Shared-revision source context; no claim that an imported test was executed.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-jsonl', type=Path, default=Path('data/research_v2_real/documents/dedup_smoke.jsonl'))
    parser.add_argument('--tokenizer', type=Path, default=Path('data/research_v1/tokenizer.json'))
    parser.add_argument('--output', type=Path, default=Path('data/research_v2_real/documents/neighbors_smoke.jsonl'))
    parser.add_argument('--edges', type=Path, default=Path('data/research_v2_real/documents/neighbors_smoke_edges.json'))
    parser.add_argument('--report', type=Path, default=Path('results/research_v2_real/neighbors_smoke.json'))
    args = parser.parse_args()
    if any(path.exists() for path in (args.output, args.edges, args.report)):
        raise FileExistsError('Neighbor fixture exists; use versioned output paths')
    with args.input_jsonl.open(encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle]
    edges, diagnostics = graph(records)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    items = list(samples(records, edges, tokenizer))
    methods = Counter(edge['method'] for edge in edges)
    roles = Counter(edge['source_role'] + '->' + edge['target_role'] for edge in edges)
    for path in (args.output, args.edges, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in items), encoding='utf-8')
    args.edges.write_text(json.dumps(edges, indent=2) + '\n', encoding='utf-8')
    report = {'schema_version': 1, 'passed': True, 'input_documents': len(records),
              'dependency_edges': len(edges), 'edge_methods': dict(methods), 'role_links': dict(roles),
              'multi_file_samples': len(items), 'tokens': sum(item['token_count'] for item in items),
              'max_sample_tokens': max((item['token_count'] for item in items), default=0),
              'input_sha256': hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest(),
              'samples_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'edges_sha256': hashlib.sha256(args.edges.read_bytes()).hexdigest(),
              'diagnostics': diagnostics,
              'scope': 'Repository context fixture. Python AST links are static imports; JS/TS/C/C++/Rust matches are conservative text candidates. No execution or complete dependency resolution. Reused file tokens must be counted separately if this lane is mixed into training.'}
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'edges': len(edges), 'samples': len(items), 'tokens': report['tokens']}))


if __name__ == '__main__':
    main()

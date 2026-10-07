from tools.repository_neighbors import graph


def record(repo, path, text, language='Python'):
    return {'repository': repo, 'revision': 'a' * 40, 'file_path': path,
            'raw_sha256': path, 'role': 'test' if path.startswith('tests/') else 'implementation',
            'language': language, 'text': f'<repo>{repo}<file>{path}\n' + text}


def test_python_absolute_and_relative_links_stay_within_repository():
    records = [record('org/a', 'src/pkg/__init__.py', ''),
               record('org/a', 'src/pkg/core.py', 'from . import helper\n'),
               record('org/a', 'src/pkg/helper.py', 'x = 1\n'),
               record('org/a', 'tests/test_core.py', 'from pkg.core import x\n'),
               record('org/b', 'src/pkg/other.py', 'x = 2\n')]
    edges, _ = graph(records)
    assert any(e['source_path'] == 'src/pkg/core.py' and e['target_path'] == 'src/pkg/helper.py' for e in edges)
    assert any(e['source_path'] == 'tests/test_core.py' and e['target_path'] == 'src/pkg/core.py' for e in edges)
    assert all(e['repository'] == 'org/a' for e in edges)


def test_relative_typescript_and_quoted_header_links():
    records = [record('org/a', 'src/main.ts', 'import { value } from "./value";', 'TypeScript'),
               record('org/a', 'src/value.ts', 'export const value = 1;', 'TypeScript'),
               record('org/a', 'src/main.cpp', '#include "thing.h"\n', 'C++'),
               record('org/a', 'include/thing.h', '#pragma once\n', 'C++')]
    edges, _ = graph(records)
    assert {e['target_path'] for e in edges} == {'src/value.ts', 'include/thing.h'}


def test_rust_named_module_resolves_children_in_its_own_directory():
    records = [record('org/a', 'src/foo.rs', 'mod bar;', 'Rust'),
               record('org/a', 'src/foo/bar.rs', 'pub fn hello() {}', 'Rust'),
               record('org/a', 'src/bar.rs', 'pub fn unrelated() {}', 'Rust')]
    edges, _ = graph(records)
    assert len(edges) == 1 and edges[0]['target_path'] == 'src/foo/bar.rs'

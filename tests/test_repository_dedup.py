import copy
import hashlib

import pytest

from tools.repository_dedup import deduplicate, family_ids


def record(repo, path, body):
    return {'repository': repo, 'revision': 'a' * 40, 'file_path': path,
            'text': f'<repo>{repo}<file>{path}\n' + body,
            'raw_sha256': hashlib.sha256(body.encode()).hexdigest(),
            'normalized_sha256': hashlib.sha256(body.rstrip().encode()).hexdigest()}


def test_global_duplicate_ignores_headers_and_line_endings():
    a = record('org/a', 'a.py', 'def hello():\n    return 42\n')
    b = record('org/b', 'b.py', 'def hello():\r\n    return 42\r\n')
    b['normalized_sha256'] = a['normalized_sha256']
    kept, removed, report = deduplicate([b, a], audit_all_pairs=True)
    assert len(kept) == 1 and removed[0]['reason'] == 'exact_normalized'
    assert report['counts']['cross_repository_removals'] == 1


def test_near_duplicate_uses_body_and_confirmed_similarity():
    body = '\n'.join(f'value_{i} = {i} + other_{i}' for i in range(120))
    a = record('org/a', 'a.py', body)
    b = record('org/b', 'b.py', body.replace('value_53', 'changed_53'))
    before = copy.deepcopy([b, a])
    kept, removed, report = deduplicate([b, a], audit_all_pairs=True)
    assert len(kept) == 1 and removed[0]['reason'] == 'near_duplicate'
    assert removed[0]['jaccard'] >= .85
    assert [b, a] == before
    assert deduplicate([a, b], audit_all_pairs=True) == (kept, removed, report)


def test_unrelated_code_is_retained():
    a = record('org/a', 'a.py', '\n'.join(f'x_{i} = {i}' for i in range(80)))
    b = record('org/b', 'b.py', '\n'.join(f'print("word_{i}", {i * 10})' for i in range(80)))
    kept, removed, _ = deduplicate([a, b], audit_all_pairs=True)
    assert len(kept) == 2 and not removed


def test_fork_families_merge_transitively_and_deterministically():
    repos = {'org/a', 'org/b', 'org/c', 'org/d'}
    assert family_ids(repos, [['org/c', 'org/b'], ['org/a', 'org/c']]) == {
        'org/a': 'org/a', 'org/b': 'org/a', 'org/c': 'org/a', 'org/d': 'org/d'}
    with pytest.raises(ValueError, match='unknown'):
        family_ids(repos, [['org/a', 'other/z']])


def test_reject_transformed_or_corrupt_input():
    a = record('org/a', 'a.py', 'source text')
    a['fim'] = {'applied': True}
    with pytest.raises(ValueError, match='before FIM'):
        deduplicate([a])
    a.pop('fim')
    a['normalized_sha256'] = 'incorrect'
    with pytest.raises(ValueError, match='hash mismatch'):
        deduplicate([a])

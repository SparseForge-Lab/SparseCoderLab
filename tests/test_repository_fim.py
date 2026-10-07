import copy
import hashlib

import pytest

from tools.repository_fim import restore, transform


def record(kind='code', language='Python', body=None):
    body = body or ('def summarize(values):\n    """Return the résumé of this collection."""\n'
                    '    total = sum(values)\n    count = len(values)\n    return total, count\n')
    return {'text': '<repo>example/project<file>src/main.py\n' + body, 'repository': 'example/project',
            'file_path': 'src/main.py', 'revision': 'a' * 40, 'kind': kind, 'language': language,
            'raw_sha256': hashlib.sha256(body.encode()).hexdigest()}


def test_fim_preserves_source_and_provenance_without_mutating_input():
    original = record()
    before = copy.deepcopy(original)
    out = transform(original, seed=42, ratio=1)
    assert out['fim']['applied']
    assert restore(out) == original['text']
    assert out['raw_sha256'] == original['raw_sha256']
    assert original == before
    assert out == transform(original, seed=42, ratio=1)
    assert out['fim']['prefix_characters'] > 0 and out['fim']['suffix_characters'] > 0


@pytest.mark.parametrize('sample', [record(kind='technical'), record(language='TOML'),
                                    record(body='<fim_prefix>' + 'a' * 200)])
def test_unsuitable_material_is_not_reordered(sample):
    out = transform(sample, seed=42, ratio=1)
    assert not out['fim']['applied']
    assert out['text'] == sample['text']


def test_zero_ratio_and_double_transform_guard():
    original = record()
    assert transform(original, seed=42, ratio=0)['text'] == original['text']
    with pytest.raises(ValueError, match='twice'):
        transform(transform(original, seed=42, ratio=1), seed=42, ratio=1)

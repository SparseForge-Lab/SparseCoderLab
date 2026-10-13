import copy
import json

import pytest

from forgebuilder.strata_runtime import derive_config, prepare


def test_only_context_and_runtime_settings_change(tmp_path):
    source = {'args': ['--native', 'D:/models/first.gguf', '--ple-gguf', 'C:/models/second.gguf',
                       '--max-context', '262144', '--kv', 'q4_0', '--mmap-experts'],
              'host': '0.0.0.0', 'port': 8095, 'log': 'original.log',
              'sampling': {'temperature': .6}, 'reasoning_budget_tokens': 8192,
              'vision': {'gpu': False}, 'model_name': 'full-iq3_xxs'}
    before = copy.deepcopy(source)
    result = derive_config(source, tmp_path)
    expected = copy.deepcopy(source)
    expected['args'][5] = '32768'
    expected.update(host='127.0.0.1', port=8096, log=str(tmp_path / 'engine.log'))
    assert result == expected
    assert source == before


def test_missing_or_ambiguous_context_rejected(tmp_path):
    for args in ([], ['--max-context', '1', '--max-context', '2']):
        with pytest.raises(ValueError):
            derive_config({'args': args}, tmp_path)


def test_source_installation_cannot_be_runtime(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'args': ['--max-context', '262144']}))
    with pytest.raises(ValueError, match='outside'):
        prepare(source, tmp_path / 'runtime')

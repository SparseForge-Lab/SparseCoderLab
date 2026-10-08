import json
import subprocess
import sys
import pytest
from pathlib import Path

from src.training.data import split_document


def test_repository_split_keeps_all_files_together():
    first = split_document('file one', 42, 0.2, 'owner/repo')
    second = split_document('different file', 42, 0.2, 'owner/repo')
    assert first == second


def test_repo_export_records_stable_revision_and_filters_secrets(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'LICENSE').write_text('MIT License\n', encoding='utf-8')
    (repo / 'app.py').write_text('def answer():\n    return 42\n', encoding='utf-8')
    (repo / 'README.md').write_text('# Example\nRun the test suite. Unicode separators: \u2028 and \u2029.\n', encoding='utf-8')
    (repo / 'Makefile').write_text('test:\n\tpython -m pytest\n', encoding='utf-8')
    (repo / 'tests').mkdir()
    (repo / 'tests' / 'test_app.py').write_text('def test_answer():\n    assert 42 == 42\n', encoding='utf-8')
    (repo / 'credentials.py').write_text('API_KEY = "this_is_a_fake_secret_value_123"\n', encoding='utf-8')
    (repo / 'apache.py').write_text('# SPDX-License-Identifier: Apache-2.0\nprint(1)\n', encoding='utf-8')
    (repo / 'held.py').write_text('# SPDX-License-Identifier: GPL-3.0-only\nprint(2)\n', encoding='utf-8')
    (repo / 'AGENTS.md').write_text('Private automation notes\n', encoding='utf-8')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 'test@example.invalid'], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'Test'], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', 'LICENSE', 'app.py', 'credentials.py', 'README.md', 'Makefile', 'tests', 'apache.py', 'held.py', 'AGENTS.md'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'fixture'], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    output = tmp_path / 'records.jsonl'
    subprocess.run([
        sys.executable, 'tools/import_local_repo.py', '--repo', str(repo), '--source-id', 'example/project',
        '--source-url', 'https://github.com/example/project', '--license', 'MIT', '--output', str(output),
    ], check=True, capture_output=True, text=True)

    with output.open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    record = next(row for row in rows if row['file_path'] == 'app.py')
    manifest = json.loads(output.with_suffix('.jsonl.manifest.json').read_text(encoding='utf-8'))
    quarantine = output.with_suffix('.jsonl.quarantine.jsonl').read_text(encoding='utf-8')
    assert record['repository'] == 'example/project'
    assert record['revision'] == revision
    assert record['raw_sha256']
    import hashlib
    committed = subprocess.check_output(['git', '-C', str(repo), 'show', 'HEAD:app.py'])
    assert record['raw_sha256'] == hashlib.sha256(committed).hexdigest()
    roles = {row['file_path']: row['role'] for row in rows}
    assert roles['README.md'] == 'documentation'
    assert roles['Makefile'] == 'build'
    assert roles['tests/test_app.py'] == 'test'
    assert next(row for row in rows if row['file_path'] == 'apache.py')['license'] == 'Apache-2.0'
    assert 'held.py' not in roles and 'AGENTS.md' not in roles
    assert manifest['filtered_counts']['unreviewed_spdx_expression'] == 1
    assert str(repo) not in json.dumps(record)
    assert manifest['revision'] == revision
    assert manifest['filtered_counts']['secret_quarantined'] == 1
    assert 'credentials.py' in quarantine
    assert 'fake_secret' not in quarantine
    from tools.repository_data_smoke import validate_export
    validated_source, validated_records = validate_export(output)
    assert validated_source['revision'] == revision and validated_records == rows
    bare = tmp_path / 'bare.git'
    subprocess.run(['git', 'clone', '--bare', str(repo), str(bare)], check=True, capture_output=True)
    bare_output = tmp_path / 'bare_records.jsonl'
    subprocess.run([
        sys.executable, 'tools/import_local_repo.py', '--repo', str(bare), '--source-id', 'example/project',
        '--source-url', 'https://github.com/example/project', '--license', 'MIT', '--output', str(bare_output),
    ], check=True, capture_output=True)
    assert not (bare / 'LICENSE').exists()
    assert bare_output.read_bytes() == output.read_bytes()
    repeated = subprocess.run([
        sys.executable, 'tools/import_local_repo.py', '--repo', str(repo), '--source-id', 'example/project',
        '--source-url', 'https://github.com/example/project', '--license', 'MIT', '--output', str(output),
    ], capture_output=True, text=True)
    assert repeated.returncode != 0
    assert 'Export already exists' in repeated.stderr


def test_batch_blobs_preserves_binary_bytes_and_validates_object_identity(tmp_path):
    from tools.import_local_repo import BatchBlobs
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    values = [b'a\r\nb\n\x00\xff', bytes(range(256))*4097, b'']
    ids = [subprocess.check_output(['git', '-C', str(tmp_path), 'hash-object', '-w', '--stdin'], input=raw).decode().strip() for raw in values]
    with BatchBlobs(tmp_path) as batch:
        for oid, raw in zip(ids, values): assert batch.read(oid) == raw
        import pytest
        with pytest.raises(ValueError, match='Invalid Git object'): batch.read('HEAD; unsafe')


@pytest.mark.parametrize('explicit_assignments', [False, True])
def test_packing_retains_repository_provenance_and_isolates_splits(tmp_path, monkeypatch, explicit_assignments):
    import hashlib
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from src.training.data import SPECIAL_TOKENS
    from tools.prepare_data import prepare

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'results').mkdir()
    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=266, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(['sample'], trainer)
    tokenizer.save('tokenizer.json')
    docs = []
    for repo in range(20):
        for file in range(2):
            text = f'def value_{repo}_{file}():\n    return {repo}\n'
            docs.append({'text': text, 'source': f'example/repo{repo}', 'repository': f'example/repo{repo}',
                         'revision': 'a' * 40, 'file_path': f'src/file{file}.py',
                         'raw_sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'license': 'MIT', 'language': 'Python', 'kind': 'code', 'role': 'implementation',
                         'repository_family': 'example/shared-family' if repo < 2 else f'example/repo{repo}',
                         'fim': {'applied': file == 1, 'eligible': True}})
    duplicate = dict(docs[0], repository='example/duplicate', source='example/duplicate')
    duplicate['text'] = '<repo>different metadata\n' + duplicate['text']
    docs.append(duplicate)
    input_path = tmp_path / 'input.jsonl'
    input_path.write_text(''.join(json.dumps(doc) + '\n' for doc in docs), encoding='utf-8')
    cfg = {'model': {'vocab_size': tokenizer.get_vocab_size()}, 'data': {
        'dataset_version': 'research_v2_smoke', 'shards': 'data/shards', 'manifest': 'results/manifest.json',
        'cache': 'data/cache', 'cache_gb': .01, 'seed': 42, 'eval_fraction': .5, 'tokenizer': 'tokenizer.json',
        'max_tokens': 100_000, 'shard_tokens': 256,
    }}
    if explicit_assignments:
        cfg['data']['repository_split_assignments'] = {doc['repository_family']: ('val' if doc['repository'].endswith(('3','7','9')) else 'train') for doc in docs}
    report = prepare(cfg, input_path)
    groups = {}
    family_groups = {}
    total_docs = 0
    for split in ('train', 'val'):
        rows = [json.loads(line) for line in Path(f'data/shards/{split}_documents.jsonl').read_text().splitlines()]
        groups[split] = {row['repository'] for row in rows}
        family_groups[split] = {row['repository_family'] for row in rows}
        assert all('fim' in row for row in rows)
        assert sum(row['length'] for row in rows) == report['tokens'][split]
        assert all(row['revision'] == 'a' * 40 and row['file_path'].startswith('src/') for row in rows)
        total_docs += len(rows)
    assert groups['train'] and groups['val'] and groups['train'].isdisjoint(groups['val'])
    assert family_groups['train'].isdisjoint(family_groups['val'])
    assert total_docs == 40
    assert report['rejected']['exact_or_normalized_duplicate'] == 1
    assert report['dataset_version'] == 'research_v2_smoke'
    if explicit_assignments:
        for split in ('train', 'val'):
            assert all(cfg['data']['repository_split_assignments'][family] == split for family in family_groups[split])
        assert report['split_policy'].startswith('Reviewed explicit')
        cfg['data'].update(shards='data/missing',cache='data/missing_cache',manifest='results/missing.json',repository_split_assignments={})
        with pytest.raises(ValueError, match='Missing or invalid reviewed'):
            prepare(cfg,input_path)


def test_repo_export_rejects_tracked_modifications(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'LICENSE').write_text('MIT License\n', encoding='utf-8')
    (repo / 'app.py').write_text('print(1)\n', encoding='utf-8')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 'test@example.invalid'], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'Test'], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', 'LICENSE', 'app.py'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'fixture'], check=True)
    (repo / 'app.py').write_text('print(2)\n', encoding='utf-8')
    result = subprocess.run([
        sys.executable, 'tools/import_local_repo.py', '--repo', str(repo), '--source-id', 'example/project',
        '--source-url', 'https://github.com/example/project', '--license', 'MIT', '--output', str(tmp_path / 'out.jsonl'),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'tracked modifications' in result.stderr

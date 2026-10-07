"""Prepare an isolated seed1337 workspace only after an explicit primary review.

Never launches training. The frozen runner is copied unchanged; only the resolved
training seed changes. Existing primary checkpoints are never copied or edited.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

from src.config import fingerprint, load_config
from src.training.research import ENDPOINTS, frozen_identity, sha, training_hash

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / 'work/replication_seed1337'
PUBLIC = ROOT / 'results/replication_seed1337'


def preflight(decision_path: Path) -> dict:
    decision = json.loads(decision_path.read_text(encoding='utf-8'))
    assert decision['required'] is True and decision['seed'] == 1337
    models = decision['models']
    assert len(models) == 2 and len(set(models)) == 2
    assert set(models) <= {'dense', 'sparse', 'memory'}
    assert decision['endpoint'] in (50003968, 100007936)
    assert decision['rationale'].strip()
    if sys.platform == 'win32':
        query = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Where-Object { $_.CommandLine -match '-m tools\\.research_train( |$)' } | "
                 "Select-Object -ExpandProperty ProcessId")
        active = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command', query], text=True)
        if active.strip():
            raise RuntimeError('Primary training is still live; replication preparation deferred')
    gate = json.loads((ROOT / 'results/research_v1/readiness_gate.json').read_text())
    assert gate['passed'] and gate['training_source_hash'] == training_hash()
    primary = {}
    for tag in ('dense', 'sparse', 'memory'):
        identity = frozen_identity(load_config(f'configs/research_v1/{tag}.yaml'))
        for endpoint in ENDPOINTS:
            summary = json.loads(Path(f'experiments/research_v1/{tag}/summary_{endpoint}.json').read_text())
            assert summary['seed'] == 42 and summary['tokens_seen'] == endpoint
            assert all(summary[k] == v for k, v in identity.items())
            assert sha(summary['checkpoint']) == summary['checkpoint_sha256']
            assert Path(f'results/research_v1/{tag}_{endpoint}_evaluation.json').is_file()
            primary[f'{tag}/{endpoint}'] = summary['checkpoint_sha256']
    integrity = json.loads((ROOT / 'results/research_v1/final_integrity.json').read_text())
    assert integrity['passed'] and len(integrity['milestones']) == 12
    return {'decision': decision, 'primary_checkpoint_hashes': primary,
            'training_source_hash': training_hash(), 'readiness_gate_sha256': sha(ROOT / 'results/research_v1/readiness_gate.json')}


def prepare(decision_path: Path) -> dict:
    proof = preflight(decision_path)
    # Refuse reuse rather than overwrite a potentially trained workspace.
    if WORKSPACE.exists() or PUBLIC.exists():
        raise RuntimeError('Replication workspace/evidence already exists; inspect it before reuse')
    WORKSPACE.mkdir(parents=True)
    PUBLIC.mkdir(parents=True)
    inventory = []
    for directory in ('src', 'tools'):
        for source in sorted((ROOT / directory).rglob('*.py')):
            if '__pycache__' in source.parts:
                continue
            relative = source.relative_to(ROOT)
            target = WORKSPACE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            assert sha(source) == sha(target)
            inventory.append({'path': relative.as_posix(), 'sha256': sha(source)})
    for filename in ('verify_install.py', 'requirements.lock'):
        shutil.copy2(ROOT / filename, WORKSPACE / filename)
        inventory.append({'path': filename, 'sha256': sha(ROOT / filename)})
    configs = {}
    for tag in ('dense', 'sparse', 'memory'):
        cfg = load_config(f'configs/research_v1/{tag}.yaml')
        original = fingerprint(cfg)
        cfg['training']['seed'] = 1337
        target = WORKSPACE / f'configs/research_v1/{tag}.yaml'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        restored = load_config(target)
        assert restored['training']['seed'] == 1337
        new_hash = fingerprint(restored)
        restored['training']['seed'] = 42
        assert fingerprint(restored) == original, 'Config changed beyond training seed'
        configs[tag] = {'seed42_config_hash': original, 'seed1337_config_hash': new_hash}
        public_config = PUBLIC / f'config_{tag}.yaml'
        shutil.copy2(target, public_config)
    data = WORKSPACE / 'data/research_v1'
    data.mkdir(parents=True)
    shutil.copy2(ROOT / 'data/research_v1/tokenizer.json', data / 'tokenizer.json')
    shard_target = ROOT / 'data/research_v1/shards'
    if sys.platform == 'win32':
        # Literal fixed paths, one shell, no filesystem deletion or moving.
        junction = data / 'shards'
        command = "New-Item -ItemType Junction -Path '" + str(junction).replace("'", "''") + "' -Target '" + str(shard_target).replace("'", "''") + "' | Out-Null"
        subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], check=True)
    else:
        (data / 'shards').symlink_to(shard_target, target_is_directory=True)
    assert (data / 'shards').resolve() == shard_target.resolve()
    results = WORKSPACE / 'results/research_v1'
    results.mkdir(parents=True)
    for filename in ('evaluation_index.json', 'corpus_manifest.json', 'readiness_gate.json'):
        shutil.copy2(ROOT / 'results/research_v1' / filename, results / filename)
        assert sha(results / filename) == sha(ROOT / 'results/research_v1' / filename)
    runtime = ROOT / '.venv/Scripts/python.exe' if sys.platform == 'win32' else ROOT / '.venv/bin/python'
    output = subprocess.check_output([str(runtime), '-c', 'from src.training.research import training_hash; print(training_hash())'], cwd=WORKSPACE, text=True)
    assert output.strip() == proof['training_source_hash'], 'Frozen source copy differs'
    with zipfile.ZipFile(PUBLIC / 'source_snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for item in inventory:
            archive.write(WORKSPACE / item['path'], item['path'])
    proof.update(status='prepared_not_trained', seed=1337, configs=configs, source_inventory=inventory,
                 workspace_project_relative='work/replication_seed1337',
                 data_access='Shared shards for consumption by unchanged verify/train/eval code only; do not run preparation tools here.',
                 seed_scope='Model/training RNG1337; frozen packed training/evaluation order42.',
                 inherited_scope_text='Unchanged runner summary says fresh seed42; classify by actual config/checkpoint seed1337 and retain this exception.',
                 source_snapshot_sha256=sha(PUBLIC / 'source_snapshot.zip'))
    (PUBLIC / 'preparation.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true', help='Create isolated workspace after every preflight passes')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run from canonical project root')
    proof = prepare(args.decision) if args.prepare else preflight(args.decision)
    print(json.dumps({'status': proof.get('status', 'preflight_only'), 'seed':1337,
                      'models':proof['decision']['models'], 'endpoint':proof['decision']['endpoint'],
                      'training_started':False}))


if __name__ == '__main__':
    main()

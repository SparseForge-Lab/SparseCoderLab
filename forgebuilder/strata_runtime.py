"""Derive an isolated dataset teacher runtime from an existing Strata setup."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def derive_config(source: dict, runtime: Path, port: int = 8096, reserve_mib: int | None = None) -> dict:
    """Inherit model settings with explicit context and optional headroom overrides."""
    if not 1024 <= port <= 65535:
        raise ValueError('invalid local port')
    result = copy.deepcopy(source)
    args = result['args']
    if args.count('--max-context') != 1:
        raise ValueError('source must define exactly one max-context argument')
    args[args.index('--max-context') + 1] = '32768'
    if reserve_mib is not None:
        if not 64 <= reserve_mib <= 16384 or args.count('--vram-reserve-mib') != 1:
            raise ValueError('a valid reserve and one source vram-reserve argument are required')
        args[args.index('--vram-reserve-mib')+1] = str(reserve_mib)
    result.update(host='127.0.0.1', port=port, log=str(runtime / 'engine.log'))
    return result


def prepare(source: Path, runtime: Path, port: int = 8096, reserve_mib: int | None = None) -> dict:
    source, runtime = source.resolve(), runtime.resolve()
    if runtime == source.parent or source.parent in runtime.parents:
        raise ValueError('runtime must be outside the source installation')
    runtime.mkdir(parents=True, exist_ok=True)
    if (runtime / 'process.json').exists():
        raise FileExistsError('runtime already has a process receipt; use a new runtime directory')
    shared = source.with_name(source.stem + '.shared-settings.json')
    config = derive_config(json.loads(source.read_text(encoding='utf8')), runtime, port, reserve_mib)
    # Keep the exact source shard arguments, including the established C: symlink.
    assets = []
    for flag in ('--native', '--ple-gguf', '--expert-profile'):
        path = Path(config['args'][config['args'].index(flag) + 1])
        if not path.is_file():
            raise FileNotFoundError(path)
        assets.append(dict(argument=flag, path=str(path), resolved_path=str(path.resolve()),
                           size_bytes=path.stat().st_size))
    server = source.parent / 'serve' / 'server.py'
    python = source.parent / '.venv' / 'Scripts' / 'python.exe'
    protected = [source, shared, server, Path(config['exe'])]
    hashes = {str(p): digest(p) for p in protected}
    target = runtime / 'flash32k.json'
    target.write_text(json.dumps(config, indent=2) + '\n', encoding='utf8')
    target.with_name('flash32k.shared-settings.json').write_bytes(shared.read_bytes())
    command = [str(python), str(server), '--engine', 'strata', '--config', str(target),
               '--host', '127.0.0.1', '--port', str(port)]
    receipt = dict(context=32768, port=port, config=str(target), config_sha256=digest(target),
                   source_hashes=hashes, assets=assets, command=command, student_training=False,
                   vram_reserve_mib=reserve_mib)
    (runtime / 'prepared.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    return receipt


def assert_sources_unchanged(receipt: dict) -> None:
    for path, expected in receipt['source_hashes'].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f'protected Strata source changed: {path}')


def start(receipt: dict, runtime: Path) -> dict:
    assert_sources_unchanged(receipt)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', receipt['port']))
    if digest(Path(receipt['config'])) != receipt['config_sha256']:
        raise RuntimeError('derived config changed after preparation')
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    with (runtime / 'server.log').open('ab') as output:
        process = subprocess.Popen(receipt['command'], cwd=str(Path(receipt['command'][1]).parent.parent),
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                   creationflags=flags)
    state = dict(pid=process.pid, started_unix=time.time(), port=receipt['port'],
                 command=receipt['command'], status='starting')
    (runtime / 'process.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf8')
    return state


def health(port: int) -> dict:
    with urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
        state = json.load(response)
    if state.get('service') != 'strata' or state.get('max_context') != 32768:
        raise RuntimeError(f'unexpected teacher runtime: {state}')
    return state


def fingerprint_weights(runtime: Path) -> dict:
    """Hash the actual shards, converted packs, MTP and control/vision assets once."""
    receipt = json.loads((runtime / 'prepared.json').read_text(encoding='utf8'))
    assert_sources_unchanged(receipt)
    config = json.loads(Path(receipt['config']).read_text(encoding='utf8'))
    args = config['args']
    paths = {Path(a['resolved_path']) for a in receipt['assets']}
    for flag in ('--pack', '--mtp'):
        directory = Path(args[args.index(flag) + 1])
        paths.update(p for p in directory.rglob('*') if p.is_file())
    if '--control-vector-scaled' in args:
        paths.add(Path(args[args.index('--control-vector-scaled')+1].rsplit(':',1)[0]))
    if config.get('vision'):
        paths.add(Path(config['vision']['mmproj']))
        paths.add(Path(config['vision']['exe']))
    rows = []
    for path in sorted(paths):
        before = path.stat()
        value = digest(path)
        after = path.stat()
        if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
            raise RuntimeError(f'model asset changed during fingerprint: {path}')
        rows.append(dict(path=str(path), size_bytes=after.st_size, sha256=value,
                         mtime_ns=after.st_mtime_ns))
        print(json.dumps({'stage':'teacher_fingerprint','path':str(path),'size_bytes':after.st_size}),flush=True)
    result = dict(status='sha256_verified', assets=rows, source_config_sha256=receipt['config_sha256'],
                  identity_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
    (runtime/'weight_identity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    assert_sources_unchanged(receipt)
    return result

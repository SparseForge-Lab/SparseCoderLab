from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

def source_hash() -> str:
    h = hashlib.sha256()
    paths = sorted(p for directory in ['src','tools','tests','configs'] for p in Path(directory).rglob('*') if p.suffix in ('.py','.yaml'))
    paths.extend([Path('verify_install.py'), Path('requirements.lock')])
    for path in paths: h.update(str(path).encode()); h.update(path.read_bytes())
    return h.hexdigest()

def run_gate() -> dict:
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '--junitxml=results/tests.xml'], capture_output=True, text=True)
    Path('results/test_output.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
    overfit = json.loads(Path('results/overfit.json').read_text()) if Path('results/overfit.json').exists() else {}
    profiles = {p.name: json.loads(p.read_text()) for p in Path('results').glob('profile_*.json')}
    required = ['profile_DenseCompute.json','profile_SparseV3.json','profile_SparseV3Ngram.json']
    passed = result.returncode == 0 and overfit.get('passed', False) and all(profiles.get(p, {}).get('passed', False) for p in required)
    passed = passed and profiles.get('profile_DenseCompute.json', {}).get('seconds', 0) >= 300
    report = {'passed': passed, 'source_hash': source_hash(), 'pytest_returncode': result.returncode,
              'overfit_passed': overfit.get('passed', False), 'required_profiles': required,
              'five_minute_dense_profiler': profiles.get('profile_DenseCompute.json', {}).get('seconds', 0) >= 300,
              'gate_scope': 'Engineering readiness only; does not authorize or automatically start training'}
    Path('results/phase0_gate.json').write_text(json.dumps(report, indent=2)); print(result.stdout); return report

if __name__ == '__main__':
    r = run_gate(); print(json.dumps(r, indent=2)); sys.exit(0 if r['passed'] else 1)

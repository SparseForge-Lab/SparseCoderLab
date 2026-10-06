import argparse, json
from pathlib import Path
from src.config import load_config
from src.training.engine import run

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--max-wall-minutes', type=float); p.add_argument('--target-tokens', type=int); p.add_argument('--max-steps', type=int)
    p.add_argument('--resume', type=Path); p.add_argument('--seed', type=int); p.add_argument('--microbatch', type=int)
    a = p.parse_args(); cfg = load_config(a.config)
    if a.seed is not None: cfg['training']['seed'] = a.seed
    if a.microbatch is not None: cfg['training']['microbatch'] = a.microbatch
    # Longer jobs are gated on completed project-local phase-0 checks, tied to source hash.
    if (a.max_wall_minutes if a.max_wall_minutes is not None else cfg['training']['max_wall_minutes']) > 30:
        from tools.phase0 import source_hash
        gate = Path('results/phase0_gate.json')
        if not gate.exists() or json.loads(gate.read_text()).get('source_hash') != source_hash():
            raise RuntimeError('Long run blocked: current source needs passing unit tests, overfit, resume parity and profiler')
        if not json.loads(gate.read_text()).get('passed'): raise RuntimeError('Phase 0 failed')
    print(json.dumps(run(cfg, a.run_dir, max_wall_minutes=a.max_wall_minutes, target_tokens=a.target_tokens,
                         max_steps=a.max_steps, resume=a.resume), indent=2))

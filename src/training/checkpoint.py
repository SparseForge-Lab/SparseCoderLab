from __future__ import annotations
import os, random
from pathlib import Path
import numpy as np
import torch

def rng_state() -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state: dict) -> None:
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda']: torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])

def atomic_save(path: Path, state: dict, project_limit: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix('.tmp')
    if project_limit is not None:
        root = Path(__file__).resolve().parents[2]
        used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        # Model+optimizer+temporary replacement allowance; fail before writing.
        estimate = sum(v.numel() * v.element_size() for v in state['model'].values()) * 4
        if used + estimate > project_limit: raise RuntimeError('Project budget would be exceeded by atomic checkpoint')
    try:
        with temp.open('wb') as f: torch.save(state, f); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally: temp.unlink(missing_ok=True)

def save_training(path: Path, model, optimizer, scheduler, metadata: dict, limit: int | None = None) -> None:
    atomic_save(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                       'rng': rng_state(), 'scaler': None, 'metadata': metadata}, limit)

def resume_training(path: Path, model, optimizer, scheduler, device: str) -> dict:
    # Only load this project's trusted local checkpoints; pickle format is not an interchange format.
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng']); return state['metadata']

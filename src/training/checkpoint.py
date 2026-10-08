from __future__ import annotations
import copy, os, random, time
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
    metadata = dict(metadata, optimizer_decay_policy=getattr(optimizer,'decay_policy','unspecified'))
    atomic_save(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                       'rng': rng_state(), 'scaler': None, 'metadata': metadata}, limit)


def load_optimizer_state(model, optimizer, saved):
    """Migrate the historical one-group order without discarding Adam moments."""
    groups = saved['param_groups']
    if len(groups) == len(optimizer.param_groups):
        for old, new in zip(groups, optimizer.param_groups):
            if 'param_names' in old and old['param_names'] != new.get('param_names'):
                raise ValueError('Optimizer parameter identity/order mismatch')
        optimizer.load_state_dict(saved)
        return False
    if len(groups) != 1 or len(optimizer.param_groups) != 2:
        raise ValueError('Unsupported optimizer group migration')
    named = list(model.named_parameters())
    original = groups[0]
    if len(named) != len(original['params']):
        raise ValueError('Historical optimizer parameter count mismatch')
    if 'param_names' in original and original['param_names'] != [n for n, _ in named]:
        raise ValueError('Historical optimizer parameter identity mismatch')
    ids = {name: index for (name, _), index in zip(named, original['params'])}
    for (_, parameter), index in zip(named, original['params']):
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key in saved['state'].get(index, {}) and saved['state'][index][key].shape != parameter.shape:
                raise ValueError('Historical optimizer tensor shape mismatch')
    migrated = dict(state=saved['state'], param_groups=[])
    for new in optimizer.param_groups:
        migrated['param_groups'].append(dict(original, params=[ids[n] for n in new['param_names']],
                                             param_names=new['param_names'], weight_decay=new['weight_decay']))
    optimizer.load_state_dict(migrated)
    return True


def load_scheduler_state(scheduler, saved, migrated=False):
    if migrated:
        saved = copy.deepcopy(saved)
        for key in ('base_lrs', '_last_lr', 'lr_lambdas'):
            if key in saved:
                if len(saved[key]) != 1:
                    raise ValueError('Historical scheduler group count mismatch')
                saved[key] = saved[key] * len(scheduler.optimizer.param_groups)
    scheduler.load_state_dict(saved)

def resume_training(path: Path, model, optimizer, scheduler, device: str) -> dict:
    # Only load this project's trusted local checkpoints; pickle format is not an interchange format.
    started = time.perf_counter()
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    migrated = load_optimizer_state(model, optimizer, state['optimizer'])
    load_scheduler_state(scheduler, state['scheduler'], migrated)
    restore_rng(state['rng'])
    metadata = dict(state['metadata'])
    metadata['checkpoint_load_seconds'] = time.perf_counter() - started
    if migrated:
        metadata['optimizer_group_migration'] = 'legacy_all_to_matrix_except_ngram_v1; moments/steps retained, decay policy corrected'
    return metadata

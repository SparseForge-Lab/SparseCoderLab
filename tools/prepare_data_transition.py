"""Create explicit new-phase states from immutable, trusted local 100M states."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import shutil
from pathlib import Path
import torch
from src.config import fingerprint, load_config
from src.model import LanguageModel
from src.training.checkpoint import atomic_save, load_optimizer_state
from src.training.engine import make_optimizer
from src.utils.hashing import sha256_file

INHERITED_TOKENS = 100_007_936
PHASE_UPDATES = 18_310
PHASE_TOKENS = 149_995_520


def transition_config(canonical, data_config):
    cfg = copy.deepcopy(canonical)
    cfg['model']['activation_checkpointing'] = True
    cfg['data'] = copy.deepcopy(data_config['data'])
    cfg['training'].update(warmup_steps=500, schedule_steps=PHASE_UPDATES,
        target_tokens=INHERITED_TOKENS+PHASE_TOKENS, optimizer_decay_policy='matrix_except_ngram_v1')
    cfg['data']['project_gb'] = 100
    # Corpus cache has its own preprocessing budget; active training storage
    # includes frozen shards and tokenizer, rather than all historical downloads.
    cfg['data']['training_storage_budget_gb'] = 6
    cfg['data']['checkpoint_free_space_reserve_gb'] = 20
    return cfg


def create_transition(cfg, entry, destination):
    if destination.exists(): raise FileExistsError('Never overwrite transition states; choose a new experiment')
    if sha256_file(Path(cfg['data']['tokenizer'])) != entry['tokenizer_sha256']:
        raise ValueError('Checkpoint tokenizer migration is not authorized')
    parent = Path(entry['full_resume_state']['path'])
    snapshot = Path(entry['model_weights']['path'])
    for path, key in ((parent,'full_resume_state'),(snapshot,'model_weights')):
        if sha256_file(path) != entry[key]['sha256']: raise ValueError('Canonical checkpoint identity mismatch')
    if shutil.disk_usage(parent).free < 20*1024**3: raise RuntimeError('Checkpoint reserve below 20 GiB')
    full = torch.load(parent, map_location='cpu', weights_only=False)
    if full['metadata']['tokens_seen'] != INHERITED_TOKENS: raise ValueError('Wrong inherited token count')
    if full['metadata']['config_hash'] != entry['config_fingerprint']:
        raise ValueError('Canonical config identity mismatch')
    if full['metadata']['data_hash'] != entry['data_manifest_sha256']:
        raise ValueError('Canonical data identity mismatch')
    weights = torch.load(snapshot, map_location='cpu', weights_only=True)
    if full['model'].keys() != weights.keys() or any(not torch.equal(v,weights[k]) for k,v in full['model'].items()):
        raise ValueError('Model-only/full-state canonical weights disagree')
    del weights
    model = LanguageModel(cfg); model.load_state_dict(full['model'],strict=True)
    optimizer, scheduler = make_optimizer(model,cfg)
    migrated = load_optimizer_state(model, optimizer, full['optimizer'])
    # Loading Adam restores its old LR. Start the new scheduler at phase step0
    # and explicitly set both groups to the new warmup rate without touching moments.
    base_lr = cfg['training']['lr']; warmup = cfg['training']['warmup_steps']
    for group in optimizer.param_groups:
        group['initial_lr'] = base_lr; group['lr'] = base_lr/max(1,warmup)
    scheduler.base_lrs = [base_lr]*len(optimizer.param_groups)
    scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
    metadata = dict(full['metadata'])
    metadata.update(config_hash=fingerprint(cfg),data_hash=sha256_file(Path(cfg['data']['shards'])/'manifest.json'),
        cursor=0,wall_time=0.,training_seconds=0.,inherited_tokens=INHERITED_TOKENS,
        phase_start_step=metadata['step'],phase_steps=0,new_phase_tokens=0,
        optimizer_decay_policy='matrix_except_ngram_v1',
        transition=dict(parent_sha256=entry['full_resume_state']['sha256'],
            weights_sha256=entry['model_weights']['sha256'],
            optimizer='Inherited Adam moments and per-parameter steps; corrected no-decay groups',
            rng='Inherited Python/NumPy/Torch/CUDA streams',
            scheduler='Reset phase-relative 500-update warmup, cosine decay through18310 updates, min_ratio0.1',
            data_cursor='Reset to0 on the new frozen document-isolated corpus',
            inherited_step=metadata['step']))
    # Construction and hash checking may consume randomness; retain exact parent
    # state in the output, including CUDA RNG when this preparation runs on CPU.
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        rng=full['rng'],scaler=None,metadata=metadata)
    atomic_save(destination,state)
    loaded=torch.load(destination,map_location='cpu',weights_only=False)
    if any(not torch.equal(v,loaded['model'][k]) for k,v in full['model'].items()):
        raise ValueError('Transition writer changed model weights')
    receipt=dict(checkpoint_path=destination.as_posix(),checkpoint_sha256=sha256_file(destination),
        checkpoint_bytes=destination.stat().st_size,config_hash=fingerprint(cfg),
        inherited_tokens=INHERITED_TOKENS,new_phase_tokens=0,phase_updates=PHASE_UPDATES,
        effective_batch_tokens=cfg['training']['microbatch']*cfg['training']['accumulation']*cfg['training']['context'],
        optimizer_groups_migrated=migrated,initial_group_lrs=[g['lr'] for g in optimizer.param_groups],
        scheduler_last_epoch=scheduler.last_epoch,transition=metadata['transition'],passed=True)
    if receipt['effective_batch_tokens'] != 8192: raise ValueError('Frozen transition batch differs')
    del loaded, state, full, model, optimizer, scheduler; gc.collect()
    return receipt


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-config',required=True)
    parser.add_argument('--run-root',type=Path,required=True); parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists() or args.run_root.exists(): raise FileExistsError('Use new transition outputs')
    manifest=json.loads(Path('results/prompt3/transition_checkpoint_manifest.json').read_text(encoding='utf8'))
    data_cfg=load_config(args.data_config); rows={}
    for tag,entry in manifest['canonical_transition_checkpoints'].items():
        if sha256_file(Path(entry['config']['path'])) != entry['config']['sha256']:
            raise ValueError('Canonical config bytes changed')
        cfg=transition_config(load_config(entry['config']['path']),data_cfg)
        run=args.run_root/tag; run.mkdir(parents=True)
        (run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf8')
        rows[tag]=create_transition(cfg,entry,run/'checkpoints'/'initial.pt')
        print(json.dumps(dict(variant=tag,passed=rows[tag]['passed'])),flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(dict(schema_version=1,passed=True,variants=rows,
        source_sha256=sha256_file(Path(__file__))),indent=2)+'\n',encoding='utf8')


if __name__=='__main__': main()

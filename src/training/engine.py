from __future__ import annotations
import contextlib, copy, csv, hashlib, json, math, random, shutil, signal, subprocess, time
from pathlib import Path
import numpy as np
import torch
from src.config import fingerprint
from src.model import LanguageModel
from src.training.checkpoint import save_training, resume_training
from src.training.data import PackedStream
from src.memory.ngram import NgramMemory
from verify_install import verify

LEADERBOARD = ['model','seed','stored_params','active_params_est','tokens','training_flops_est','wall_time','tok_s','val_loss','code_val_loss','general_val_loss','mtp_acceptance','compaction_score','router_entropy','cache_hit_sim','bytes_per_token_sim','vram_peak']

def read_jsonl_utf8(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def require_cuda(device: str) -> None:
    if device != 'cuda': raise RuntimeError('Training requires CUDA; explicit CPU use is only supported by unit tests')
    verify()

def amp(cfg: dict): return torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['training']['bf16'])

def optimizer_groups(model, weight_decay):
    """Decay matrices except Ngram lookup tables; exclude every vector/bias."""
    tables = {id(table.weight) for module in model.modules()
              if isinstance(module, NgramMemory)
              for table in module.tables}
    decay, no_decay = [], []
    decay_names, no_decay_names = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or id(parameter) in tables:
            no_decay.append(parameter); no_decay_names.append(name)
        else:
            decay.append(parameter); decay_names.append(name)
    return [dict(params=decay, param_names=decay_names, weight_decay=weight_decay),
            dict(params=no_decay, param_names=no_decay_names, weight_decay=0.0)]


def make_optimizer(model: LanguageModel, cfg: dict):
    t = cfg['training']
    from src.training.moe_optimizer import GroupedAdamW
    grouped = any(getattr(module,'backend',None)=='grouped' for module in model.modules())
    cls = GroupedAdamW if grouped else torch.optim.AdamW
    policy = t.get('optimizer_decay_policy', 'matrix_except_ngram_v1')
    if policy not in ('legacy_all', 'matrix_except_ngram_v1'):
        raise ValueError('Unknown optimizer decay policy')
    parameters = (list(model.parameters()) if policy == 'legacy_all' else optimizer_groups(model, t['weight_decay']))
    optimizer = (cls(model, param_groups=parameters, lr=t['lr'], betas=tuple(t['betas']),
                     weight_decay=t['weight_decay'], fused=t['fused_optimizer']) if grouped else
                 cls(parameters, lr=t['lr'], betas=tuple(t['betas']), weight_decay=t['weight_decay'], fused=t['fused_optimizer']))
    optimizer.decay_policy = policy
    def rate(step: int) -> float:
        if step < t['warmup_steps']: return (step + 1) / max(t['warmup_steps'], 1)
        fraction = min((step - t['warmup_steps']) / max(t['schedule_steps'] - t['warmup_steps'], 1), 1)
        return t['min_lr_ratio'] + (1 - t['min_lr_ratio']) * (1 + math.cos(math.pi * fraction)) / 2
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, rate)

@torch.no_grad()
def validate(model: LanguageModel, cfg: dict) -> dict:
    model.eval(); t = cfg['training']; stream = PackedStream(Path(cfg['data']['shards']), 'val', t['context'], cfg['data']['seed'])
    weighted_loss = prediction_count = 0
    for _ in range(t['eval_batches']):
        x, y = stream.next(t['microbatch'], 'cuda')
        with amp(cfg): value = float(model(x, y, return_outputs=False, segment_ids=stream.last_segment_ids)['lm_loss'])
        weighted_loss += value * stream.last_prediction_count; prediction_count += stream.last_prediction_count
    loss = weighted_loss / prediction_count; result = {'val_loss': loss, 'bits_per_token': loss / math.log(2), 'validation_predictions': prediction_count}
    from src.eval.sampling import sample_documents
    with (Path(cfg['data']['shards']) / 'val_documents.jsonl').open(encoding='utf8') as handle:
        docs = sample_documents((json.loads(line) for line in handle),t['eval_batches'],t['context'],cfg['data']['seed'],lambda row:row['kind'])
    for kind in ('code', 'general'):
        weighted = tokens = 0
        for doc in docs.get(kind,[]):
            offset=doc.get('evaluation_offset',0)
            start = doc['start']+offset; length = min(doc['length']-offset, t['context'] + 1)
            data = torch.tensor(np.array(stream.tokens[start:start + length], dtype=np.int64), device='cuda')[None]
            if length < 2: continue
            with amp(cfg): value = model(data[:, :-1], data[:, 1:], return_outputs=False)['lm_loss']
            weighted += float(value) * (length - 1); tokens += length - 1
        result[f'{kind}_val_loss'] = weighted / tokens if tokens else None
    model.train(); return result

def run(cfg: dict, run_dir: Path, *, max_wall_minutes: float | None = None, target_tokens: int | None = None,
        max_steps: int | None = None, resume: Path | None = None) -> dict:
    require_cuda(cfg['training']['device']); t = cfg['training']; seed_all(t['seed'])
    from tools.shard_data import verify_shards
    verify_shards(cfg)
    if 'training_storage_budget_gb' in cfg['data']:
        used_data=sum(p.stat().st_size for p in Path(cfg['data']['shards']).rglob('*') if p.is_file())
        used_data+=Path(cfg['data']['tokenizer']).stat().st_size
        data_budget=cfg['data']['training_storage_budget_gb']
    else:
        used_data=sum(p.stat().st_size for p in Path('data').rglob('*') if p.is_file())
        data_budget=cfg['data']['cache_gb']
    if used_data>data_budget*1024**3: raise RuntimeError('Data budget exceeded')
    if t['compile']: raise RuntimeError('Compile is disabled pending separate eager parity/benchmark lane')
    torch.backends.cuda.matmul.allow_tf32 = t['tf32']; torch.backends.cudnn.allow_tf32 = t['tf32']
    model = LanguageModel(cfg).cuda(); optimizer, scheduler = make_optimizer(model, cfg)
    stream = PackedStream(Path(cfg['data']['shards']), 'train', t['context'], cfg['data']['seed'])
    manifest = Path(cfg['data']['shards']) / 'manifest.json'
    data_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    tokenizer_hash = hashlib.sha256(Path(cfg['data']['tokenizer']).read_bytes()).hexdigest()
    meta = {'step': 0, 'tokens_seen': 0, 'cursor': 0, 'wall_time': 0.0, 'training_seconds': 0.0, 'config_hash': fingerprint(cfg),
            'data_hash': data_hash, 'tokenizer_hash': tokenizer_hash, 'seed': t['seed']}
    if resume:
        meta = resume_training(resume, model, optimizer, scheduler, 'cuda')
        if (meta['config_hash'], meta['data_hash'], meta['tokenizer_hash']) != (fingerprint(cfg), data_hash, tokenizer_hash):
            raise RuntimeError('Resume config/data/tokenizer differ from checkpoint')
        stream.cursor = meta['cursor']
        prior_summary=resume.parent.parent/'summary.json'
        if prior_summary.exists():
            prior=json.loads(prior_summary.read_text())
            if prior.get('config_hash')==meta['config_hash'] and prior.get('step')==meta['step']:
                meta['wall_time']=prior['wall_time']
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    try: commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError: commit = 'uncommitted'
    meta['git_commit'] = commit
    from tools.phase0 import source_hash
    meta['source_hash']=source_hash()
    meta['git_dirty']=subprocess.run(['git','diff','--quiet'],capture_output=True).returncode!=0
    (run_dir / 'provenance.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    stop = [False]
    def stop_handler(signum, frame): stop[0] = True
    previous_handler = signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, 'SIGTERM'): previous_term = signal.signal(signal.SIGTERM, stop_handler)
    wall_limit = 60 * (t['max_wall_minutes'] if max_wall_minutes is None else max_wall_minutes)
    if wall_limit <= 0: raise ValueError('wall cap must be positive')
    token_limit = t['target_tokens'] if target_tokens is None else target_tokens
    reserve = min(t['reserve_seconds'], wall_limit * 0.1); start = time.perf_counter(); base_wall = meta['wall_time']; initial_steps = meta['step']
    torch.cuda.reset_peak_memory_stats(); measured_steps = 0; last_loss = None; norm = None
    checkpoint = run_dir / 'checkpoints' / 'last.pt'
    def save_checkpoint():
        reserve=cfg['data'].get('checkpoint_free_space_reserve_gb')
        if reserve is not None:
            estimate=sum(v.numel()*v.element_size() for v in model.state_dict().values())*4
            if shutil.disk_usage(run_dir).free < reserve*1024**3+estimate:
                raise RuntimeError('Atomic checkpoint would breach the free-space reserve')
        save_training(checkpoint,model,optimizer,scheduler,meta,
            None if reserve is not None else int(cfg['data']['project_gb']*1024**3))
    log_file = (run_dir / 'metrics.jsonl').open('a', encoding='utf-8')
    try:
        model.train()
        while meta['tokens_seen'] < token_limit and not stop[0]:
            if time.perf_counter() - start >= wall_limit - reserve: break
            if max_steps is not None and meta['step'] - initial_steps >= max_steps: break
            step_start = time.perf_counter(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.0; data_wait_seconds = 0.0
            for _ in range(t['accumulation']):
                data_start = time.perf_counter(); x, y = stream.next(t['microbatch'], 'cuda')
                data_wait_seconds += time.perf_counter() - data_start
                with amp(cfg):
                    out = model(x, y, mtp_tokens=x, return_outputs=False, segment_ids=stream.last_segment_ids); loss = out['loss'] / t['accumulation']
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss; do not advance phases')
                loss.backward(); loss_sum += float(out['lm_loss'].detach()); meta['tokens_seen'] += x.numel()
                mtp_metrics = out.get('mtp_metrics', {}); del out, loss
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'], error_if_nonfinite=True))
            memory_diagnostics = model.memory.diagnostics(x,stream.last_segment_ids) if model.memory is not None and (meta['step'] + 1) % t['log_steps'] == 0 else None
            opt_start = torch.cuda.Event(enable_timing=True); opt_end = torch.cuda.Event(enable_timing=True)
            opt_start.record(); optimizer.step(); opt_end.record(); scheduler.step(); torch.cuda.synchronize()
            elapsed = time.perf_counter() - step_start; measured_steps += 1; meta['training_seconds'] += elapsed
            meta['step'] += 1; meta['cursor'] = stream.cursor; meta['wall_time'] = base_wall + time.perf_counter() - start
            if 'inherited_tokens' in meta:
                meta['new_phase_tokens'] = meta['tokens_seen'] - meta['inherited_tokens']
                meta['phase_steps'] = meta['step'] - meta['phase_start_step']
            last_loss = loss_sum / t['accumulation']
            if meta['step'] % t['log_steps'] == 0:
                routes = model.routes(); record = dict(meta, train_loss=last_loss, grad_norm=norm,
                    tok_s=t['context'] * t['microbatch'] * t['accumulation'] / elapsed,
                    step_seconds=elapsed, data_wait_wall_seconds=data_wait_seconds,
                    optimizer_cuda_seconds=opt_start.elapsed_time(opt_end)/1000,
                    vram_peak=torch.cuda.max_memory_allocated(), vram_peak_reserved=torch.cuda.max_memory_reserved(),
                    vram_allocated=torch.cuda.memory_allocated(), vram_reserved=torch.cuda.memory_reserved(), gpu_utilization=read_utilization(),
                    router={i: {'entropy': float(r['entropy']), 'imbalance': float(r['imbalance']), 'load': r['load'].tolist()} for i, r in routes.items()},
                    mtp=mtp_metrics, ngram=memory_diagnostics)
                log_file.write(json.dumps(record) + '\n'); log_file.flush()
            if meta['step'] % t['eval_steps'] == 0 and time.perf_counter() - start < wall_limit - reserve:
                val = validate(model, cfg); log_file.write(json.dumps(dict(meta, **val)) + '\n'); log_file.flush()
            if meta['step'] % t['checkpoint_steps'] == 0:
                save_checkpoint()
        meta['wall_time'] = base_wall + time.perf_counter() - start; meta['cursor'] = stream.cursor
        save_checkpoint()
        val = validate(model, cfg) if time.perf_counter() - start < wall_limit - reserve / 2 else {'val_loss': None}
        from tools.count_params import count_model
        count = count_model(model); measured_tokens=meta.get('new_phase_tokens',meta['tokens_seen'])
        step_tok_s = measured_tokens / max(meta['training_seconds'], 1e-9)
        meta['wall_time_at_checkpoint']=meta['wall_time']
        meta['wall_time']=base_wall+time.perf_counter()-start
        tok_s=measured_tokens/max(meta['wall_time'],1e-9)
        flop_equivalent=count['active_estimate']
        if model.mtp is not None:
            flop_equivalent=count['base_active_without_mtp']+cfg['mtp']['horizons']*(count['mtp']+model.embedding.weight.numel())
        result = dict(meta, **val, model=cfg['name'], stored_params=count['total'], active_params_est=count['active_estimate'],
            tok_s=tok_s, train_step_tok_s=step_tok_s, measured_steps=measured_steps, train_loss=last_loss, grad_norm=norm,
            training_flops_est=6 * flop_equivalent * meta['tokens_seen'],
            training_flops_caveat='6N reference estimate includes routed active FFNs, repeated MTP block and additional MTP output heads; ignores attention T^2, shortened horizons, lookup arithmetic and dispatch. Not measured FLOPs.',
            vram_peak=torch.cuda.max_memory_allocated(), vram_peak_reserved=torch.cuda.max_memory_reserved(), checkpoint=str(checkpoint), resumed=resume is not None,
            throughput_scope='tok_s includes loop/logging/final checkpoint/validation after model initialization; train_step_tok_s excludes non-step overhead.',
            throughput_token_scope='New phase tokens only' if 'inherited_tokens' in meta else 'Lifetime tokens',
            comparison_basis='Same frozen tokenizer/data order, optimizer, seed, context; throughput evidence does not establish model quality.')
        if 'inherited_tokens' in meta:
            result['new_phase_training_flops_est']=6*flop_equivalent*meta['new_phase_tokens']
        (run_dir / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        if t.get('publish_leaderboard',True):
            leaderboard = Path('results/leaderboard.csv'); exists = leaderboard.exists()
            with leaderboard.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=LEADERBOARD);
                if not exists: writer.writeheader()
                row = {k: result.get(k, '') for k in LEADERBOARD}; row['tokens'] = result['tokens_seen']; writer.writerow(row)
        return result
    finally:
        log_file.close(); signal.signal(signal.SIGINT, previous_handler)
        if hasattr(signal, 'SIGTERM'): signal.signal(signal.SIGTERM, previous_term)

def read_utilization() -> int | None:
    try: return int(subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=2).splitlines()[0])
    except Exception: return None

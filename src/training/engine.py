from __future__ import annotations
import contextlib, copy, csv, hashlib, json, math, random, signal, subprocess, time
from pathlib import Path
import numpy as np
import torch
from src.config import fingerprint
from src.model import LanguageModel
from src.training.checkpoint import save_training, resume_training
from src.training.data import PackedStream
from verify_install import verify

LEADERBOARD = ['model','seed','stored_params','active_params_est','tokens','training_flops_est','wall_time','tok_s','val_loss','code_val_loss','general_val_loss','mtp_acceptance','compaction_score','router_entropy','cache_hit_sim','bytes_per_token_sim','vram_peak']

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def require_cuda(device: str) -> None:
    if device != 'cuda': raise RuntimeError('Training requires CUDA; explicit CPU use is only supported by unit tests')
    verify()

def amp(cfg: dict): return torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['training']['bf16'])

def make_optimizer(model: LanguageModel, cfg: dict):
    t = cfg['training']
    from src.training.moe_optimizer import GroupedAdamW
    grouped = any(getattr(module,'backend',None)=='grouped' for module in model.modules())
    cls = GroupedAdamW if grouped else torch.optim.AdamW
    parameters = model if grouped else model.parameters()
    optimizer = cls(parameters, lr=t['lr'], betas=tuple(t['betas']), weight_decay=t['weight_decay'], fused=t['fused_optimizer'])
    def rate(step: int) -> float:
        if step < t['warmup_steps']: return (step + 1) / max(t['warmup_steps'], 1)
        fraction = min((step - t['warmup_steps']) / max(t['schedule_steps'] - t['warmup_steps'], 1), 1)
        return t['min_lr_ratio'] + (1 - t['min_lr_ratio']) * (1 + math.cos(math.pi * fraction)) / 2
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, rate)

@torch.no_grad()
def validate(model: LanguageModel, cfg: dict) -> dict:
    model.eval(); t = cfg['training']; stream = PackedStream(Path(cfg['data']['shards']), 'val', t['context'], cfg['data']['seed'])
    values = []
    for _ in range(t['eval_batches']):
        x, y = stream.next(t['microbatch'], 'cuda')
        with amp(cfg): values.append(float(model(x, y)['lm_loss']))
    loss = sum(values) / len(values); result = {'val_loss': loss, 'bits_per_token': loss / math.log(2)}
    docs = [json.loads(line) for line in (Path(cfg['data']['shards']) / 'val_documents.jsonl').read_text().splitlines()]
    for kind in ('code', 'general'):
        weighted = tokens = 0
        for doc in [d for d in docs if d['kind'] == kind][:t['eval_batches']]:
            start = doc['start']; length = min(doc['length'], t['context'] + 1)
            data = torch.tensor(np.array(stream.tokens[start:start + length], dtype=np.int64), device='cuda')[None]
            if length < 2: continue
            with amp(cfg): value = model(data[:, :-1], data[:, 1:])['lm_loss']
            weighted += float(value) * (length - 1); tokens += length - 1
        result[f'{kind}_val_loss'] = weighted / tokens if tokens else None
    model.train(); return result

def run(cfg: dict, run_dir: Path, *, max_wall_minutes: float | None = None, target_tokens: int | None = None,
        max_steps: int | None = None, resume: Path | None = None) -> dict:
    require_cuda(cfg['training']['device']); t = cfg['training']; seed_all(t['seed'])
    from tools.shard_data import verify_shards
    verify_shards(cfg)
    used_data=sum(p.stat().st_size for p in Path('data').rglob('*') if p.is_file())
    if used_data>cfg['data']['cache_gb']*1024**3: raise RuntimeError('Data budget exceeded')
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
    torch.cuda.reset_peak_memory_stats(); measured = []; last_loss = None; norm = None
    checkpoint = run_dir / 'checkpoints' / 'last.pt'
    log_file = (run_dir / 'metrics.jsonl').open('a', encoding='utf-8')
    try:
        model.train()
        while meta['tokens_seen'] < token_limit and not stop[0]:
            if time.perf_counter() - start >= wall_limit - reserve: break
            if max_steps is not None and meta['step'] - initial_steps >= max_steps: break
            step_start = time.perf_counter(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.0
            for _ in range(t['accumulation']):
                x, y = stream.next(t['microbatch'], 'cuda')
                with amp(cfg):
                    out = model(x, y, mtp_tokens=x); loss = out['loss'] / t['accumulation']
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss; do not advance phases')
                loss.backward(); loss_sum += float(out['lm_loss'].detach()); meta['tokens_seen'] += x.numel()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'], error_if_nonfinite=True))
            memory_diagnostics = model.memory.diagnostics(x) if model.memory is not None and (meta['step'] + 1) % t['log_steps'] == 0 else None
            optimizer.step(); scheduler.step(); torch.cuda.synchronize()
            elapsed = time.perf_counter() - step_start; measured.append(elapsed); meta['training_seconds'] += elapsed
            meta['step'] += 1; meta['cursor'] = stream.cursor; meta['wall_time'] = base_wall + time.perf_counter() - start
            last_loss = loss_sum / t['accumulation']
            if meta['step'] % t['log_steps'] == 0:
                routes = model.routes(); record = dict(meta, train_loss=last_loss, grad_norm=norm,
                    tok_s=t['context'] * t['microbatch'] * t['accumulation'] / elapsed,
                    vram_peak=torch.cuda.max_memory_allocated(), gpu_utilization=read_utilization(),
                    router={i: {'entropy': float(r['entropy']), 'imbalance': float(r['imbalance']), 'load': r['load'].tolist()} for i, r in routes.items()},
                    mtp=out.get('mtp_metrics', {}), ngram=memory_diagnostics)
                log_file.write(json.dumps(record) + '\n'); log_file.flush()
            if meta['step'] % t['eval_steps'] == 0 and time.perf_counter() - start < wall_limit - reserve:
                val = validate(model, cfg); log_file.write(json.dumps(dict(meta, **val)) + '\n'); log_file.flush()
            if meta['step'] % t['checkpoint_steps'] == 0:
                save_training(checkpoint, model, optimizer, scheduler, meta, int(cfg['data']['project_gb'] * 1024**3))
        meta['wall_time'] = base_wall + time.perf_counter() - start; meta['cursor'] = stream.cursor
        save_training(checkpoint, model, optimizer, scheduler, meta, int(cfg['data']['project_gb'] * 1024**3))
        val = validate(model, cfg) if time.perf_counter() - start < wall_limit - reserve / 2 else {'val_loss': None}
        from tools.count_params import count_model
        count = count_model(model); step_tok_s = meta['tokens_seen'] / max(meta['training_seconds'], 1e-9)
        meta['wall_time_at_checkpoint']=meta['wall_time']
        meta['wall_time']=base_wall+time.perf_counter()-start
        tok_s=meta['tokens_seen']/max(meta['wall_time'],1e-9)
        flop_equivalent=count['active_estimate']
        if model.mtp is not None:
            flop_equivalent=count['base_active_without_mtp']+cfg['mtp']['horizons']*(count['mtp']+model.embedding.weight.numel())
        result = dict(meta, **val, model=cfg['name'], stored_params=count['total'], active_params_est=count['active_estimate'],
            tok_s=tok_s, train_step_tok_s=step_tok_s, measured_steps=len(measured), train_loss=last_loss, grad_norm=norm,
            training_flops_est=6 * flop_equivalent * meta['tokens_seen'],
            training_flops_caveat='6N reference estimate includes routed active FFNs, repeated MTP block and additional MTP output heads; ignores attention T^2, shortened horizons, lookup arithmetic and dispatch. Not measured FLOPs.',
            vram_peak=torch.cuda.max_memory_allocated(), checkpoint=str(checkpoint), resumed=resume is not None,
            throughput_scope='tok_s includes loop/logging/final checkpoint/validation after model initialization; train_step_tok_s excludes non-step overhead.',
            comparison_basis='Same frozen tokenizer/data order, optimizer, seed, context. Synthetic engineering smoke, not model ranking.')
        (run_dir / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
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

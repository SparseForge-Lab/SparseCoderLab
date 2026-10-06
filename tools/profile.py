from __future__ import annotations
import argparse, copy, hashlib, json, signal, time
from pathlib import Path
import torch
from src.config import load_config
from src.model import LanguageModel
from src.training.data import PackedStream
from src.training.engine import require_cuda, seed_all, amp
from src.training.checkpoint import save_training, resume_training
from src.config import fingerprint

def profile(cfg: dict, seconds: float, batch_find: bool = True, resume: Path | None = None, output: Path | None = None) -> dict:
    require_cuda('cuda'); seed_all(cfg['training']['seed']); t = cfg['training']; model = LanguageModel(cfg).cuda()
    from tools.shard_data import verify_shards
    verify_shards(cfg)
    data_hash=hashlib.sha256((Path(cfg['data']['shards'])/'manifest.json').read_bytes()).hexdigest()
    stream = PackedStream(Path(cfg['data']['shards']), 'train', t['context'], cfg['data']['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['lr'], betas=tuple(t['betas']), weight_decay=t['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    meta = {'cursor':0, 'tokens_seen':0, 'step':0, 'config_hash':fingerprint(cfg),'data_hash':data_hash}
    if resume:
        meta = resume_training(resume, model, optimizer, scheduler, 'cuda')
        if meta['config_hash'] != fingerprint(cfg): raise ValueError('Profile resume config mismatch')
        if meta['data_hash'] != data_hash:raise ValueError('Profile resume data mismatch')
        stream.cursor = meta['cursor']; batch_find=False
    stop=[False]; previous=signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0,True))
    measurements = []
    def step(batch: int) -> float:
        start = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        x, y = stream.next(batch, 'cuda')
        with amp(cfg): out = model(x, y, mtp_tokens=x)
        if not torch.isfinite(out['loss']): raise FloatingPointError('Profile NaN')
        out['loss'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'], error_if_nonfinite=True); optimizer.step(); scheduler.step()
        meta['cursor']=stream.cursor; meta['tokens_seen']+=x.numel(); meta['step']+=1
        torch.cuda.synchronize(); return time.perf_counter() - start
    if batch_find:
        for batch in t['batch_candidates']:
            if stop[0]: break
            try:
                torch.cuda.reset_peak_memory_stats(); step(batch); elapsed = sum(step(batch) for _ in range(3))
                peak = torch.cuda.max_memory_allocated(); free, total = torch.cuda.mem_get_info()
                safe = free / total >= t['headroom_fraction']
                measurements.append({'microbatch': batch, 'tok_s': batch*t['context']*3/elapsed, 'vram_peak': peak, 'safe_headroom': safe})
                if not safe: break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); measurements.append({'microbatch': batch, 'oom': True}); break
        safe = [m for m in measurements if m.get('safe_headroom')]
        if not safe: raise RuntimeError('No safe microbatch found')
        # Finder recommendations do not silently alter fairness configs.
        recommended = max(safe, key=lambda m: m['tok_s'])['microbatch']
    else: recommended = t['microbatch']
    batch = t['microbatch']; step(batch); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter(); steps = 0; durations = []
    while time.perf_counter() - start < seconds and not stop[0]:
        durations.append(step(batch)); steps += 1
    elapsed = time.perf_counter() - start
    report = {'model': cfg['name'], 'context': t['context'], 'microbatch': batch, 'seconds': elapsed, 'steps': steps,
              'tokens': steps * batch * t['context'], 'tok_s': steps * batch * t['context'] / elapsed,
              'vram_peak': torch.cuda.max_memory_allocated(), 'vram_reserved': torch.cuda.max_memory_reserved(),
              'batch_candidates': measurements, 'recommended_microbatch': recommended, 'passed': bool(steps),
              'purpose': 'Synthetic-corpus eager BF16 train-step profiler; single microbatch optimizer steps. Not gradient-accumulated end-to-end run or real-code validation.',
              'durations_first_10': durations[:10], 'max_step_seconds': max(durations) if durations else None,
              'checkpoint':f'experiments/profile_{cfg["name"]}/checkpoints/last.pt', 'paused':stop[0], 'resumed':resume is not None}
    save_training(Path(report['checkpoint']),model,optimizer,scheduler,meta,int(cfg['data']['project_gb']*1024**3))
    signal.signal(signal.SIGINT,previous)
    (output or Path(f'results/profile_{cfg["name"]}.json')).write_text(json.dumps(report, indent=2)); return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/dense_compute.yaml'); p.add_argument('--seconds', type=float, default=300); p.add_argument('--no-batch-find', action='store_true')
    p.add_argument('--resume',type=Path); p.add_argument('--output',type=Path)
    a = p.parse_args(); print(json.dumps(profile(load_config(a.config), a.seconds, not a.no_batch_find,a.resume,a.output), indent=2))

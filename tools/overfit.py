from __future__ import annotations
import argparse, json, signal, time
from pathlib import Path
import torch
from src.config import load_config
from src.model import LanguageModel
from src.training.engine import require_cuda, seed_all, amp
from src.training.checkpoint import save_training,resume_training
from src.config import fingerprint

def overfit(config: str, steps: int = 80, resume: Path | None = None, max_wall_minutes: float = 5) -> dict:
    cfg = load_config(config); require_cuda('cuda'); seed_all(cfg['training']['seed'])
    model = LanguageModel(cfg).cuda(); optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
    meta={'step':0,'losses':[],'config_hash':fingerprint(cfg),'tokens_seen':0,'cursor':0}
    if resume:
        meta=resume_training(resume,model,optimizer,scheduler,'cuda')
        if meta['config_hash']!=fingerprint(cfg): raise ValueError('Overfit config mismatch')
    stop=[False]; previous=signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    # Repeated language-like sequence, no target leakage and full-size DenseCompute backbone.
    x = torch.tensor([[11, 29, 47, 83, 101, 53, 17, 5] * 8], device='cuda'); y = torch.roll(x, -1, 1)
    losses = meta['losses']; start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for _ in range(meta['step'],steps):
        if stop[0] or time.perf_counter()-start>=max_wall_minutes*60: break
        optimizer.zero_grad()
        with amp(cfg): result = model(x, y); loss = result['loss']
        if not torch.isfinite(loss): raise FloatingPointError('overfit nonfinite')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'], error_if_nonfinite=True); optimizer.step(); scheduler.step()
        losses.append(float(result['lm_loss'].detach())); meta['step']+=1; meta['tokens_seen']+=x.numel()
    save_training(Path('experiments/overfit/checkpoints/last.pt'),model,optimizer,scheduler,meta,int(cfg['data']['project_gb']*1024**3)); signal.signal(signal.SIGINT,previous)
    complete=meta['step']>=steps
    torch.cuda.synchronize(); report = {'config': config, 'steps': meta['step'], 'requested_steps':steps,
        'initial_loss': losses[0] if losses else None, 'final_loss': losses[-1] if losses else None,
        'losses': losses, 'passed': bool(losses) and complete and losses[-1] < losses[0] * .3,
        'complete':complete, 'paused':not complete,'checkpoint':'experiments/overfit/checkpoints/last.pt',
        'seconds': time.perf_counter() - start, 'vram_peak': torch.cuda.max_memory_allocated()}
    Path('results/overfit.json').write_text(json.dumps(report, indent=2));
    if complete and not report['passed']: raise AssertionError(f'Overfit failed: {report}')
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/dense_compute.yaml'); p.add_argument('--steps', type=int, default=80)
    p.add_argument('--resume',type=Path); p.add_argument('--max-wall-minutes',type=float,default=5)
    a = p.parse_args(); print(json.dumps(overfit(a.config, a.steps,a.resume,a.max_wall_minutes), indent=2))

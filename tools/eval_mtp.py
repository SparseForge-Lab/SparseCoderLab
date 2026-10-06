import argparse, json
from pathlib import Path
import torch
from src.config import load_config
from src.model import LanguageModel
from src.training.data import PackedStream
from src.training.engine import require_cuda, amp
from src.eval.speculation import greedy, speculative_greedy

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/sparse_mtp.yaml'); p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--decode-tokens', type=int, default=16); a = p.parse_args(); cfg = load_config(a.config); require_cuda('cuda')
    model = LanguageModel(cfg).cuda().eval(); model.load_state_dict(torch.load(a.checkpoint, map_location='cuda', weights_only=False)['model'])
    stream = PackedStream(Path(cfg['data']['shards']), 'val', cfg['training']['context'], cfg['data']['seed']); x, y = stream.next(1,'cuda')
    with torch.no_grad(), amp(cfg): out = model(x,y,mtp_tokens=x)
    prefix = x[:, :32]; baseline, baseline_s = greedy(model,prefix,a.decode_tokens,cfg); sequence, metrics = speculative_greedy(model,prefix,a.decode_tokens,cfg)
    assert torch.equal(baseline,sequence)
    report = {'horizons':out['mtp_metrics'], 'speculation':metrics, 'greedy_tokens_s':a.decode_tokens/baseline_s,
              'net_speedup':baseline_s/metrics['wall_seconds'], 'greedy_parity':True, 'checkpoint':str(a.checkpoint),
              'scope':'Synthetic held-out prefix; full-prefix reference implementation. Smoke checkpoint results cannot justify keeping MTP.'}
    Path('results/mtp_evaluation.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

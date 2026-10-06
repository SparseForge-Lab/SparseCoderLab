import json
from pathlib import Path
import torch
from src.config import load_config
from src.model import LanguageModel
from src.training.engine import validate, require_cuda

require_cuda('cuda')
cfg = load_config('configs/phase1a/memory.yaml')
model = LanguageModel(cfg).cuda()
state = torch.load('experiments/phase1a_memory/checkpoints/last.pt', map_location='cuda', weights_only=False)
model.load_state_dict(state['model'])
normal = validate(model, cfg)
try:
    model.memory.ablate = True
    ablated = validate(model, cfg)
finally:
    model.memory.ablate = False
report = {'normal': normal, 'ablated': ablated,
          'delta_ablated_minus_normal': {k: ablated[k]-normal[k] for k in normal if normal[k] is not None},
          'checkpoint_tokens': state['metadata']['tokens_seen'],
          'scope': 'Same frozen checkpoint and deterministic validation samples, no retraining; synthetic single-seed evidence.'}
Path('results/phase1a_ngram_ablation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

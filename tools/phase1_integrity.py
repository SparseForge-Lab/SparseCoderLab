import hashlib, json, math
from pathlib import Path
from src.config import load_config
from tools.shard_data import verify_shards

configs = [load_config('configs/phase1a/' + tag + '.yaml') for tag in ('dense','sparse','memory')]
assert all(c['training'] == configs[0]['training'] and c['data'] == configs[0]['data'] for c in configs)
assert all(not c['mtp']['enabled'] and c['model']['top_k'] == 1 for c in configs)
attention = ('vocab_size','d_model','layers','q_heads','kv_heads','head_dim','norm_eps','rope_theta','init_std','activation_checkpointing')
assert all(all(c['model'][key] == configs[0]['model'][key] for key in attention) for c in configs)
verify_shards(configs[0])
summaries = [json.loads(Path('experiments','phase1a_' + tag,'summary.json').read_text()) for tag in ('dense','sparse','memory')]
assert all(s['tokens_seen'] == 5005312 and s['step'] == 611 and s['cursor'] == 4888 and not s['resumed'] for s in summaries)
for tag in ('dense','sparse','memory'):
    logs = [json.loads(line) for line in Path('experiments','phase1a_'+tag,'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs if 'val_loss' in r] == [200,400,600]
for key in ('source_hash','data_hash','tokenizer_hash','seed'):
    assert len({s[key] for s in summaries}) == 1
report = {'passed': True, 'common_tokens': 5005312, 'common_steps': 611, 'common_cursor': 4888,
          'fresh_initializations': True, 'matching_training_and_data_configs': True,
          'matching_attention': True, 'shard_checksums_verified': True,
          'tokenizer_sha256': hashlib.sha256(Path('data/tokenizer.json').read_bytes()).hexdigest(),
          'evaluation_steps': [200,400,600,611], 'scope': 'Deterministic token order follows matching seeded PackedStream/cursor, tested in Phase 0. Same seed does not mean identical shared initial tensors across different parameter-registration layouts.'}
Path('results/phase1a_integrity.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

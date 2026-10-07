"""Export small comparison tables from CPU-verified replication evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.training.research import sha

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'results/replication_seed1337'
KEYS = ('val_loss', 'code_val_loss', 'general_val_loss', 'technical_val_loss')


def write_table(name, rows):
    if not rows:
        return
    with (PUBLIC / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    evidence = PUBLIC / 'verification.json'
    report = json.loads(evidence.read_text(encoding='utf-8'))
    losses, deltas, ablations, routes = [], [], [], []
    for item in report['completed'].values():
        assert item['integrity']['passed']
        summary = item['summary']
        tokens, tag = summary['tokens_seen'], summary['tag']
        primary = json.loads((ROOT / f'experiments/research_v1/{tag}/summary_{tokens}.json').read_text())
        for seed, source in ((42, primary), (1337, summary)):
            assert source['seed'] == seed and source['tokens_seen'] == tokens
            losses.append({'model': tag, 'seed': seed, 'tokens': tokens,
                           **{key: source[key] for key in KEYS},
                           'checkpoint_sha256': source['checkpoint_sha256']})
        if item['ablation'] is not None:
            ablations.append({'model': tag, 'seed': 1337, 'tokens': tokens,
                              **{key: item['ablation'][key] for key in KEYS}})
        for layer, values in item['routing'].items():
            routes.append({'model': tag, 'seed': 1337, 'tokens': tokens, 'layer': layer,
                           **{key: values[key] for key in ('pooled_load_cv', 'max_share', 'unused_experts',
                                                          'category_expert_mutual_information_bits')}})
    for tokens, item in report['comparisons'].items():
        for seed in (42, 1337):
            deltas.append({'pair': item['pair'], 'seed': seed, 'tokens': int(tokens),
                           **item[f'seed{seed}_deltas']})
    for name, rows in (('loss_curves.csv', losses), ('paired_deltas.csv', deltas),
                       ('memory_ablations.csv', ablations), ('router_health.csv', routes)):
        write_table(name, rows)
    receipt = {'status': report['status'], 'verification_sha256': sha(evidence),
               'outputs': {name: sha(PUBLIC / name) for name in
                           ('loss_curves.csv', 'paired_deltas.csv', 'memory_ablations.csv', 'router_health.csv')
                           if (PUBLIC / name).exists()},
               'rows': {'losses': len(losses), 'paired_deltas': len(deltas),
                        'memory_ablations': len(ablations), 'router_health': len(routes)},
               'scope': 'Only CPU-verified seed1337 milestones and their matched seed42 references. '
                        'Signed deltas are memory minus sparse; positive means memory has higher loss. '
                        'Two training seeds do not establish a population confidence interval.'}
    (PUBLIC / 'tables_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

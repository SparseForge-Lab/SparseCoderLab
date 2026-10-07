"""CPU figures using only verified, matched two-seed replication tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.training.research import sha

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'results/replication_seed1337'
METRICS = (('val_loss', 'Mixed'), ('code_val_loss', 'Code'),
           ('general_val_loss', 'General'), ('technical_val_loss', 'Technical'))


def rows(name):
    with (PUBLIC / name).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    verification = PUBLIC / 'verification.json'
    receipt = json.loads((PUBLIC / 'tables_receipt.json').read_text())
    assert receipt['verification_sha256'] == sha(verification), 'Re-export stale tables first'
    assert all(sha(PUBLIC / name) == digest for name, digest in receipt['outputs'].items()), 'Table bytes changed'
    table = rows('loss_curves.csv')
    deltas = rows('paired_deltas.csv')
    assert len(table) == receipt['rows']['losses'] and len(deltas) == receipt['rows']['paired_deltas']
    assert deltas, 'No matched seed1337 checkpoint available'
    complete = receipt['status'] == 'verified_complete'
    outputs = []
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        for extension in ('png', 'svg', 'pdf'):
            path = PUBLIC / f'{name}.{extension}'
            fig.savefig(path, dpi=180, bbox_inches='tight')
            outputs.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for ax, (key, label) in zip(axes.flat, METRICS):
        for seed, color in ((42, '#2878b5'), (1337, '#b04c42')):
            selected = sorted((row for row in deltas if int(row['seed']) == seed), key=lambda row: int(row['tokens']))
            ax.plot([int(row['tokens']) / 1e6 for row in selected], [float(row[key]) for row in selected],
                    marker='o', color=color, label=f'Training seed {seed}')
        ax.axhline(0, color='black', linewidth=.8)
        ax.set(title=f'{label} loss difference', xlabel='Consumed tokens (millions)',
               ylabel='Ngram minus Sparse (nats/token)', xlim=(15, 105))
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle(('Complete' if complete else 'Partial') + ' selected-pair replication · matched checkpoints only\n'
                 'Negative favors Ngram; positive favors Sparse · fixed data/evaluation order 42', fontsize=12)
    save(fig, 'paired_seed_gaps')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for ax, (key, label) in zip(axes.flat, METRICS):
        for tag, color, name in (('sparse', '#e08d22', 'Sparse'), ('memory', '#218b65', 'Sparse + Ngram')):
            for seed, linestyle, marker in ((42, '-', 'o'), (1337, '--', 's')):
                selected = sorted((row for row in table if row['model'] == tag and int(row['seed']) == seed),
                                  key=lambda row: int(row['tokens']))
                if selected:
                    ax.plot([int(row['tokens']) / 1e6 for row in selected], [float(row[key]) for row in selected],
                            color=color, linestyle=linestyle, marker=marker, label=f'{name}, seed {seed}')
        ax.set(title=f'{label} validation NLL', xlabel='Consumed tokens (millions)',
               ylabel='nats/token', xlim=(15, 105))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(('Complete' if complete else 'Partial') + ' verified replication coverage\n'
                 'Seed42 references shown only where seed1337 model checkpoints are verified', fontsize=12)
    save(fig, 'paired_seed_losses')
    result = {'status': receipt['status'], 'verification_sha256': sha(verification),
              'inputs': {name: sha(PUBLIC / name) for name in ('loss_curves.csv', 'paired_deltas.csv', 'tables_receipt.json')},
              'outputs': outputs, 'visual_review': 'pending',
              'scope': 'Measured checkpoint markers only; lines connect measurements and do not imply intermediate evaluation. '
                       'Two seeds are a sensitivity check, not a population interval.'}
    (PUBLIC / 'plots_receipt.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'files': len(outputs)}))


if __name__ == '__main__':
    main()

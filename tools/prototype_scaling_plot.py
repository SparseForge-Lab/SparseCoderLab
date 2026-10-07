"""Plot measured prototype-corpus losses without implying matched later gates."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.hashsalt'] = 'prototype-scaling-v1'


def main():
    source = Path('results/prompt3/milestone_history.csv')
    with source.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    models = ['Dense', 'Sparse', 'Sparse + 10M Ngram', 'Sparse + 25M Ngram']
    palette = ['#356ac3', '#cf7950', '#208d77', '#8757b5']
    markers = ['o', 's', '^', 'D']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True)
    for model, color, marker in zip(models, palette, markers):
        points = sorted((row for row in rows if row['model'] == model), key=lambda row: int(row['tokens']))
        for axis, metric in zip(axes, ('mixed_nll', 'code_nll')):
            valid = [row for row in points if row[metric]]
            axis.plot([int(row['tokens']) / 1e6 for row in valid], [float(row[metric]) for row in valid],
                      marker=marker, color=color, label=model, linewidth=1.8, markersize=6)
    for axis, title in zip(axes, ('Mixed validation NLL', 'Code validation NLL')):
        axis.set_title(title)
        axis.set_xlabel('Cumulative training tokens (millions)')
        axis.set_ylabel('NLL (lower is better)')
        axis.set_xticks([100, 180, 250])
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Prototype corpus scaling', fontsize=14)
    fig.text(.5, .015, 'All models share the 100M comparison. Later endpoints are exploratory and unmatched.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    directory = Path('results/prompt3/figures')
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / 'prototype_scaling.png', dpi=170)
    svg = directory / 'prototype_scaling.svg'
    fig.savefig(svg, metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)


if __name__ == '__main__':
    main()

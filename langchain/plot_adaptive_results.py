"""Plot repeated N/B/O experiment summaries; no LLM or GPU required."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_adaptive_results import analyze


METRICS = [('goodput', 'Successful requests / second (higher is better)'),
           ('p50_latency', 'P50 total latency / seconds (lower is better)'),
           ('p90_latency', 'P90 total latency / seconds (lower is better)'),
           ('makespan', 'Makespan / seconds (lower is better)'),
           ('failed', 'Failed requests (lower is better)'),
           ('preparation_client_overlap_ratio', 'Preparation/client wall overlap ratio')]


def plot(root, output=None):
    root = Path(root)
    aggregate, effects = analyze(root)
    output = Path(output) if output else root / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    trials = defaultdict(dict)
    for path in sorted(root.glob('*/summary.json')):
        repeat = int(path.parent.name.split('_', 1)[0])
        row = json.loads(path.read_text(encoding='utf-8'))
        if repeat in trials[row['mode']]:
            raise ValueError(f'duplicate repeat {repeat}: {row["mode"]}')
        trials[row['mode']][repeat] = row
    modes = sorted(trials)
    stats = {(r['mode'], r['metric']): r for r in aggregate}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})

    def save(fig, name):
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Repeated trials: means, sample SD and individual trial values', fontsize=15)
    for ax, (metric, label) in zip(axes.flat, METRICS):
        available = [mode for mode in modes if stats[mode, metric]['mean'] is not None]
        if not available:
            ax.text(.5, .5, 'Not collected', ha='center', va='center', transform=ax.transAxes)
        for x, mode in enumerate(available):
            stat = stats[mode, metric]
            ax.bar(x, stat['mean'], color='#3b82b6' if mode.startswith('fixed') else '#e58b3b',
                   alpha=.7, width=.6)
            if stat['stdev'] is not None:
                ax.errorbar(x, stat['mean'], yerr=stat['stdev'], fmt='none', color='black', capsize=4)
            values = [r[metric] for r in trials[mode].values() if r.get(metric) is not None]
            xs = [x + .18 * (i / max(1, len(values)-1) - .5) for i in range(len(values))]
            ax.scatter(xs, values, s=23, color='black', zorder=3)
            ax.text(x, .98, f'n={len(values)}', ha='center', va='top', transform=ax.get_xaxis_transform())
        ax.set_xticks(range(len(available)), available, rotation=20)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2)
    save(fig, 'comparison')

    for family in ('n', 'b', 'o'):
        baseline, dynamic = f'fixed-{family}', f'dynamic-{family}'
        if baseline not in trials or dynamic not in trials:
            continue
        pairs = sorted(trials[baseline].keys() & trials[dynamic].keys())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle(f'{family.upper()}: paired repeats (each line is one repeat)')
        for ax, (metric, label) in zip(axes, METRICS[:3]):
            for repeat in pairs:
                values = [trials[mode][repeat].get(metric) for mode in (baseline, dynamic)]
                if all(value is not None for value in values):
                    ax.plot([0, 1], values, 'o-', alpha=.65, label=f'Repeat {repeat}')
            ax.set_xticks([0, 1], [baseline, dynamic])
            ax.set_title(label, fontsize=9)
            ax.grid(axis='y', alpha=.2)
        if len(pairs) <= 10:
            axes[-1].legend(fontsize=8)
        save(fig, f'paired_{family}')

    # Confidence intervals are resampled across trial pairs, not requests.
    for metric, label in METRICS[:4]:
        selected = [r for r in effects if r['metric'] == metric]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle('Paired differences: dynamic minus fixed')
        for y, row in enumerate(selected):
            mean = row['mean_difference']
            if row['ci95_low'] is not None:
                ax.plot([row['ci95_low'], row['ci95_high']], [y, y], color='#3b82b6', linewidth=3)
            ax.plot(mean, y, 'o', color='black')
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_yticks(range(len(selected)),
                      [f'{r["dynamic"]} ({r["pairs"]} pairs)' for r in selected])
        ax.set_xlabel(label + '\nLines: paired bootstrap 95% CI; one pair has no CI')
        ax.set_ylim(-.6, len(selected)-.4)
        save(fig, f'effect_{metric}')
    print(f'Charts saved to {output.resolve()}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='Directory produced by run_adaptive_experiments.py')
    parser.add_argument('--output-dir', type=Path, help='Default: RESULTS/figures')
    args = parser.parse_args()
    try:
        plot(args.results, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

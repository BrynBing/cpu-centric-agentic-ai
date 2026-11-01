#!/usr/bin/env python3
"""Plot Fixed-B vs Dynamic-B benchmark results.

Place this file in the cpu-centric-agentic-ai repository root and run:
    python plot_scheduler_results.py
Or supply a different results folder:
    python plot_scheduler_results.py --results-dir /path/to/results

Requires: pandas, numpy, matplotlib
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')  # works on headless SSH servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODES = ['fixed-b', 'dynamic-b']
COLORS = {'fixed-b': '#64748b', 'dynamic-b': '#0f766e'}
METRICS = {
    'makespan': ('Makespan', 'Seconds', 'lower'),
    'goodput': ('Goodput', 'Jobs / second', 'higher'),
    'p50_latency': ('P50 end-to-end latency', 'Seconds', 'lower'),
    'p90_latency': ('P90 end-to-end latency', 'Seconds', 'lower'),
    'controller_cpu_seconds': ('Controller CPU time', 'CPU seconds', 'lower'),
    'preparation_client_overlap_ratio': ('Preparation-client overlap', 'Percent', 'higher'),
}


def style():
    plt.rcParams.update({
        'figure.dpi': 130, 'savefig.dpi': 180,
        'font.size': 10, 'axes.titlesize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.18,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })


def save(fig, folder, filename):
    fig.savefig(folder / filename, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', folder / filename)


def mean_comparison(agg, out):
    metrics = list(METRICS)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics):
        title, unit, _ = METRICS[metric]
        sub = agg[agg['metric'].eq(metric)].set_index('mode')
        vals = [float(sub.loc[m, 'mean']) if m in sub.index else np.nan for m in MODES]
        stds = [float(sub.loc[m, 'stdev']) if m in sub.index else np.nan for m in MODES]
        scale = 100 if metric.endswith('_ratio') else 1
        values = np.asarray(vals) * scale
        errors = np.asarray(stds) * scale
        bars = ax.bar(['Fixed-B', 'Dynamic-B'], values, yerr=errors, capsize=4,
                      color=[COLORS[m] for m in MODES], width=0.55, alpha=0.88)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y')
        ax.grid(axis='x', visible=False)
        for bar, val in zip(bars, values):
            if np.isfinite(val):
                ax.annotate(f'{val:.4f}' if unit == 'Jobs / second' else f'{val:.2f}',
                            (bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
    fig.suptitle('Fixed-B vs Dynamic-B | mean ± sample standard deviation (not CI)', fontsize=14)
    save(fig, out, '01_mean_comparison.png')


def repeat_lines(df, out):
    metrics = ['makespan', 'goodput', 'p50_latency', 'p90_latency']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics):
        title, unit, _ = METRICS[metric]
        for mode in MODES:
            ds = df.loc[df['mode'].eq(mode)].sort_values('repeat')
            ax.plot(ds['repeat'], ds[metric], '-o', markersize=6, linewidth=2,
                    label=mode, color=COLORS[mode])
        ax.set_title(title)
        ax.set_xlabel('Repeat')
        ax.set_ylabel(unit)
        ax.set_xticks(sorted(df['repeat'].dropna().unique()))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.055), frameon=False)
    fig.suptitle('Metrics by paired repeat', y=1.105, fontsize=14)
    save(fig, out, '02_repeat_comparison.png')


def paired_differences(df, out):
    metrics = ['makespan', 'goodput', 'p50_latency', 'p90_latency']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    a = df[df['mode'].eq('fixed-b')].set_index('repeat')
    b = df[df['mode'].eq('dynamic-b')].set_index('repeat')
    common = a.index.intersection(b.index).sort_values()
    if len(common) == 0:
        print('Warning: no matching repeat IDs; skipping paired differences')
        plt.close(fig)
        return
    for ax, metric in zip(axes.flat, metrics):
        title, unit, _ = METRICS[metric]
        diffs = b.loc[common, metric].astype(float) - a.loc[common, metric].astype(float)
        barcolors = ['#0f766e' if v < 0 else '#b45309' for v in diffs] if metric != 'goodput' \
            else ['#0f766e' if v > 0 else '#b45309' for v in diffs]
        ax.bar([str(x) for x in common], diffs, color=barcolors, width=0.58)
        ax.axhline(0, color='#222222', linewidth=0.9)
        ax.axhline(diffs.mean(), color='#7c3aed', linestyle='--', linewidth=1.3,
                   label=f'Mean: {diffs.mean():+.4f}')
        ax.set_title(title)
        ax.set_xlabel('Repeat')
        ax.set_ylabel(f'Dynamic-B minus Fixed-B ({unit})')
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Paired differences | negative is better except for goodput', fontsize=13)
    save(fig, out, '03_paired_differences.png')


def effect_intervals(effects, out):
    # Display raw reported bounds; do not recompute or call them statistically validated.
    measures = [m for m in METRICS if m in effects['metric'].values]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), constrained_layout=True)
    for ax, group in zip(axes, [measures[:4], measures[4:]]):
        for idx, metric in enumerate(group):
            d = effects[effects['metric'].eq(metric)].iloc[0]
            mul = 100 if metric.endswith('_ratio') else 1
            mid, low, high = (float(d[k]) * mul for k in ('mean_difference', 'ci95_low', 'ci95_high'))
            ax.errorbar(mid, idx, xerr=[[mid-low], [high-mid]], fmt='o', color=COLORS['dynamic-b'],
                        ecolor='#475569', capsize=4, markersize=6)
        ax.set_yticks(range(len(group)))
        ax.set_yticklabels([METRICS[m][0] + (' (pp)' if m.endswith('_ratio') else '') for m in group])
        ax.invert_yaxis()
        ax.axvline(0, color='#b45309', linestyle='--', linewidth=1)
        ax.set_xlabel('Reported mean difference and interval (Dynamic-B − Fixed-B)')
    fig.suptitle('Reported paired effects and supplied CI bounds (only 3 pairs; interpret cautiously)', fontsize=12)
    save(fig, out, '04_reported_effect_intervals.png')


def other_metrics(df, out):
    metrics = ['preparation_client_overlap_ratio', 'controller_cpu_seconds', 'mean_cpu_util', 'p90_service']
    names = ['Preparation-client overlap (%)', 'Controller CPU seconds', 'Mean CPU utilization (%)', 'P90 service (s)']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, metric, name in zip(axes.flat, metrics, names):
        factor = 100 if metric in ('preparation_client_overlap_ratio', 'mean_cpu_util') else 1
        for mode in MODES:
            d = df[df['mode'].eq(mode)].sort_values('repeat')
            if metric not in d or d[metric].notna().sum() == 0:
                continue
            ax.plot(d['repeat'], d[metric] * factor, '-o', label=mode, color=COLORS[mode])
        ax.set_title(name)
        ax.set_xlabel('Repeat')
        ax.set_xticks(sorted(df['repeat'].dropna().unique()))
    axes.flat[0].legend(frameon=False)
    fig.suptitle('Auxiliary metrics | overlap here is not measured hardware CPU/GPU overlap', fontsize=12)
    save(fig, out, '05_auxiliary_metrics.png')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent / 'results',
                   help='Directory containing aggregate.csv, comparison.csv, paired_effects.csv')
    args = p.parse_args()
    folder = args.results_dir.expanduser().resolve()
    files = {name: folder / f'{name}.csv' for name in ('aggregate', 'comparison', 'paired_effects')}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        sys.exit('Missing CSV file(s):\n  ' + '\n  '.join(missing))
    agg, comp, effects = (pd.read_csv(files[k]) for k in files)
    required = {
        'aggregate': {'mode', 'metric', 'mean', 'stdev'},
        'comparison': {'mode', 'repeat', *METRICS.keys()},
        'paired_effects': {'metric', 'mean_difference', 'ci95_low', 'ci95_high'},
    }
    for name, frame in [('aggregate', agg), ('comparison', comp), ('paired_effects', effects)]:
        absent = required[name] - set(frame.columns)
        if absent:
            sys.exit(f'{name}.csv missing columns: {sorted(absent)}')
    present = set(comp['mode'].dropna())
    if not set(MODES).issubset(present):
        sys.exit('comparison.csv must contain both fixed-b and dynamic-b modes')
    out = folder / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    style()
    mean_comparison(agg, out)
    repeat_lines(comp, out)
    paired_differences(comp, out)
    effect_intervals(effects, out)
    other_metrics(comp, out)
    unavailable = [c for c in ('gpu_idle_seconds', 'actual_cpu_gpu_overlap_seconds', 'mean_gpu_util')
                   if c in comp and comp[c].notna().sum() == 0]
    if unavailable:
        print('Note: GPU metrics missing (NOT zero):', ', '.join(unavailable))
    print('Done. All plots saved in:', out)


if __name__ == '__main__':
    main()

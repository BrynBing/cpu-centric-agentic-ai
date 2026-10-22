"""Aggregate independent trials and bootstrap paired treatment differences.

Positive goodput differences favor dynamic scheduling; negative latency/makespan
differences favor it. Intervals resample trial pairs, not correlated requests.
"""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from scheduling.metrics import percentile


METRICS = ('goodput', 'p50_latency', 'p90_latency', 'makespan', 'failed',
           'controller_cpu_seconds', 'preparation_client_overlap_ratio')


def analyze(root, samples=2000, seed=42):
    if samples < 1:
        raise ValueError('bootstrap sample count must be positive')
    rows = []
    for path in sorted(root.glob('*/summary.json')):
        repeat, _ = path.parent.name.split('_', 1)
        result = json.loads(path.read_text(encoding='utf-8'))
        config = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))
        shared_config = {k: v for k, v in config['config'].items() if k != 'mode'}
        rows.append(dict(repeat=int(repeat), digest=config['requests_sha256'],
                         shared_config=shared_config, **result))
    if not rows:
        raise ValueError('no trial summaries found')
    if any(row['aborted'] for row in rows):
        raise ValueError('aborted trials present; inspect them before comparative analysis')
    groups = defaultdict(list)
    for row in rows:
        groups[row['mode']].append(row)
    aggregate = []
    for mode, trials in sorted(groups.items()):
        for metric in METRICS:
            values = [trial[metric] for trial in trials if trial[metric] is not None]
            aggregate.append(dict(mode=mode, metric=metric, trials=len(values),
                                  mean=statistics.mean(values) if values else None,
                                  stdev=statistics.stdev(values) if len(values) > 1 else None))
    rng = random.Random(seed)
    effects = []
    for baseline, dynamic in [('fixed-b', 'dynamic-b'), ('fixed-o', 'dynamic-o')]:
        before = {r['repeat']: r for r in groups[baseline]}
        after = {r['repeat']: r for r in groups[dynamic]}
        pairs = [(before[k], after[k]) for k in sorted(before.keys() & after.keys())]
        if any(a['digest'] != b['digest'] for a, b in pairs):
            raise ValueError('paired trials have different request manifests')
        if any(a['shared_config'] != b['shared_config'] for a, b in pairs):
            raise ValueError('paired trials have different fixed configurations')
        for metric in METRICS:
            differences = [b[metric] - a[metric] for a, b in pairs
                           if a[metric] is not None and b[metric] is not None]
            if not differences:
                continue
            draws = [statistics.mean(rng.choices(differences, k=len(differences)))
                     for _ in range(samples)] if len(differences) >= 2 else []
            effects.append(dict(baseline=baseline, dynamic=dynamic, metric=metric,
                                pairs=len(differences), mean_difference=statistics.mean(differences),
                                ci95_low=percentile(draws, .025), ci95_high=percentile(draws, .975)))
    for name, data in [('aggregate.csv', aggregate), ('paired_effects.csv', effects)]:
        if data:
            with (root / name).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader(); writer.writerows(data)
    return aggregate, effects


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    analyze(args.results)
    print('Wrote aggregate.csv and paired_effects.csv where matched pairs exist.')

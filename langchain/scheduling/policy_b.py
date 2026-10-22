"""Dynamic B: cohort membership with a fixed admission cap."""
import math
from dataclasses import dataclass

from .metrics import percentile


@dataclass
class Estimate:
    throughput: float
    latency: float
    count: int = 1
    age: int = 0


class BatchPolicy:
    """Adjacent online exploration with EWMA scores, dwell and CPU hysteresis.

    Cohort barriers are fixed by the runtime. Partial cohorts do not train targets.
    Safety reductions take effect only on the next cohort, never running work.
    """

    def __init__(self, grid, initial, latency_target=30.0, dwell=3,
                 confirmations=2, probe_every=5, min_samples=2, expiry=20):
        self.grid = sorted(set(grid))
        if not self.grid or self.grid[0] < 1 or initial not in self.grid:
            raise ValueError('positive batch grid must contain initial batch size')
        if latency_target <= 0 or min(dwell, confirmations, probe_every,
                                      min_samples, expiry) < 1:
            raise ValueError('policy intervals and latency target must be positive')
        self.target = initial
        self.latency_target = latency_target
        self.dwell, self.confirmations = dwell, confirmations
        self.probe_every, self.min_samples, self.expiry = probe_every, min_samples, expiry
        self.estimates = {}
        self.reference = None
        self.since_change = self.since_probe = 0
        self.recommendation, self.confirmed = None, 0
        self.cpu = None
        self.high = self.low = 0
        self.saturated = False
        self.failed = False

    def sample_cpu(self, value):
        if value is None:
            return
        if not 0 <= value <= 1:
            raise ValueError('CPU utilization must be a fraction')
        self.cpu = value if self.cpu is None else .8 * self.cpu + .2 * value
        self.high = self.high + 1 if self.cpu > .85 else 0
        self.low = self.low + 1 if self.cpu < .65 else 0
        self.saturated = self.high >= 3 or value > .95

    def observe(self, target, actual, duration, successful_latencies, failed):
        if actual != target:
            return
        self.failed = bool(failed)
        for estimate in self.estimates.values():
            estimate.age += 1
        self.since_change += 1
        self.since_probe += 1
        if not successful_latencies:
            # Never learn an artificially fast latency from failed requests.
            self.estimates.pop(target, None)
            return
        x = len(successful_latencies) / max(duration, 1e-9)
        latency = percentile(successful_latencies, .9)
        old = self.estimates.get(target)
        if old is None or old.age > self.expiry:
            self.estimates[target] = Estimate(x, latency)
        else:
            self.estimates[target] = Estimate(.7 * old.throughput + .3 * x,
                                               .7 * old.latency + .3 * latency,
                                               old.count + 1)
        if self.reference is None:
            self.reference = max(x, 1e-9)

    def choose(self, pending, oldest_age=0):
        index = self.grid.index(self.target)
        if self.saturated:
            return self._switch(self.grid[max(0, index - 1)], 'cpu_pressure')
        current = self.estimates.get(self.target)
        if self.failed:
            self.confirmed = 0
            return self.target, 'failure_hold'
        if current is None or current.count < self.min_samples or current.age > self.expiry:
            self.confirmed = 0
            return self.target, 'learn_current'
        if oldest_age + current.latency >= self.latency_target:
            self.confirmed = 0
            return self.target, 'latency_urgent'
        if self.since_change < self.dwell:
            self.confirmed = 0
            return self.target, 'dwell'
        neighbors = self.grid[max(0, index - 1):index + 2]
        valid = [b for b in neighbors if b in self.estimates
                 and self.estimates[b].count >= self.min_samples
                 and self.estimates[b].age <= self.expiry]
        if self.since_probe >= self.probe_every:
            for b in neighbors:
                # Unknown CPU does not license an upward resource probe.
                if b != self.target and b <= pending and b not in valid:
                    if b < self.target or self.low >= 3:
                        self.since_probe = 0
                        return self._switch(b, 'probe')
        def score(b):
            est = self.estimates[b]
            return (est.throughput / self.reference
                    - .25 * est.latency / self.latency_target
                    - .02 * abs(math.log2(b / self.target)))
        best = max(valid, key=lambda b: (score(b), b == self.target, -b))
        if best == self.target or score(best) - score(self.target) <= .05:
            self.confirmed = 0
            self.recommendation = None
            return self.target, 'hold'
        self.confirmed = self.confirmed + 1 if best == self.recommendation else 1
        self.recommendation = best
        if self.confirmed >= self.confirmations:
            return self._switch(best, 'score')
        return self.target, 'confirm'

    def _switch(self, target, reason):
        if target != self.target:
            self.since_change = 0
            self.confirmed = 0
            self.recommendation = None
        self.target = target
        return target, reason



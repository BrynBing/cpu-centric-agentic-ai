"""Dynamic O: release timing with fixed cohort size and capacity."""
from collections import deque


class OverlapPolicy:
    """Predict remaining CLIENT cohort drain; never claim GPU kernel time."""

    def __init__(self, safety=.05, max_hold=5.0, history=32, min_samples=4):
        if safety < 0 or max_hold <= 0 or min_samples < 1 or history < min_samples:
            raise ValueError('invalid overlap predictor bounds')
        self.safety, self.max_hold = safety, max_hold
        self.min_samples = min_samples
        self.preparation = deque(maxlen=history)
        self.inference = deque(maxlen=history)

    def observe(self, preparation=None, inference=None):
        if preparation is not None:
            self.preparation.append(preparation)
        if inference is not None:
            self.inference.append(inference)

    def delay(self, elapsed, held):
        if elapsed is None:
            return 0.0, 'idle', None, None
        residuals = [g - elapsed for g in self.inference if g > elapsed]
        if len(self.preparation) < self.min_samples or len(residuals) < self.min_samples:
            return 0.0, 'cold_or_tail_fallback', None, None
        def envelope(samples, sign):
            mean = sum(samples) / len(samples)
            mad = sum(abs(v - mean) for v in samples) / len(samples)
            return max(0.0, mean + sign * mad)
        upper_p = envelope(self.preparation, 1)
        lower_r = envelope(residuals, -1)
        delay = min(max(0.0, lower_r - upper_p - self.safety),
                    max(0.0, self.max_hold - held))
        return delay, 'prediction' if delay else 'due', lower_r, upper_p

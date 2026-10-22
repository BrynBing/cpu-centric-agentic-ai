"""Dynamic N: the existing nonpreemptive process-admission update rule.

CPU inputs and thresholds are percentages (0..100), unlike B's fractions.
Sampling, process management and admission remain the runtime's responsibility.
The strict comparisons and one-sided bounds preserve the legacy behavior,
including holding at either threshold; no new clamping or smoothing is added.
"""


def update_concurrency(current, cpu_percent, minimum=4, maximum=64, step=4,
                       low_cpu=65.0, high_cpu=85.0):
    """Return the next admission target without terminating existing work."""
    if cpu_percent > high_cpu:
        return max(minimum, current - step)
    if cpu_percent < low_cpu:
        return min(maximum, current + step)
    return current

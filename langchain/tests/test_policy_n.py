"""Regression coverage for the extracted legacy N rule and policy imports."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scheduling.policy_n import update_concurrency


class ConcurrencyPolicyTest(unittest.TestCase):
    def test_thresholds_steps_and_bounds(self):
        cases = [
            (16, 64.9, 20), (16, 65, 16), (16, 75, 16),
            (16, 85, 16), (16, 85.1, 12),
            (4, 99, 4), (64, 0, 64), (6, 99, 4), (62, 0, 64),
        ]
        for current, cpu, expected in cases:
            with self.subTest(current=current, cpu=cpu):
                self.assertEqual(update_concurrency(current, cpu), expected)

    def test_custom_settings_and_no_new_initial_clamping(self):
        self.assertEqual(update_concurrency(10, 20, minimum=2, maximum=12,
                                            step=3, low_cpu=30, high_cpu=70), 12)
        self.assertEqual(update_concurrency(10, 80, minimum=2, maximum=12,
                                            step=3, low_cpu=30, high_cpu=70), 7)
        # Preserve legacy semantics even for an out-of-bounds initial value.
        self.assertEqual(update_concurrency(100, 75), 100)
        self.assertEqual(update_concurrency(100, 90), 96)

    def test_compatibility_exports_refer_to_split_implementations(self):
        import adaptive_policies as old
        from scheduling.policy_b import BatchPolicy, Estimate
        from scheduling.policy_o import OverlapPolicy
        from scheduling.metrics import percentile
        self.assertIs(old.BatchPolicy, BatchPolicy)
        self.assertIs(old.Estimate, Estimate)
        self.assertIs(old.OverlapPolicy, OverlapPolicy)
        self.assertIs(old.percentile, percentile)


if __name__ == '__main__':
    unittest.main()

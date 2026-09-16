from __future__ import annotations

import unittest

from rl.metrics import add_reward_metrics


class _Batch:
    non_tensor_batch = {
        "uid": ["a", "a", "b", "b"],
        "score": [1.0, -1.0, -1.05, -1.05],
        "strict_success": [1.0, 0.0, 0.0, 0.0],
        "attr_score": [1.0, 0.0, 0.0, 0.0],
        "option_score": [1.0, 0.0, 0.0, 0.0],
        "price_ok": [1.0, 0.0, 0.0, 0.0],
        "outcome": ["correct_purchase", "invalid_model", "timeout", "timeout"],
        "termination": ["purchase", "invalid_model", "context_limit", "context_limit"],
        "invalid_count": [0, 1, 0, 0],
    }


class MetricsTests(unittest.TestCase):
    def test_reward_components_and_group_variance_are_scalars(self):
        metrics = add_reward_metrics({}, _Batch())
        self.assertEqual(metrics["reward/success_rate"], 0.25)
        self.assertEqual(metrics["reward/outcome/timeout_rate"], 0.5)
        self.assertEqual(metrics["reward/termination/context_limit_rate"], 0.5)
        self.assertGreater(metrics["reward/group_score_std_mean"], 0.0)
        self.assertEqual(metrics["reward/group_zero_variance_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from rl.reward import compute_score, compute_trajectory_reward, evaluate_compliance, strict_type_match


GOAL = {
    "asin": "target",
    "category": "家装›灯具›投光灯",
    "attributes": ["50瓦", "防水"],
    "goal_options": ["高亮50W"],
    "price_upper": 44.0,
}
CORRECT = {
    "asin": "target",
    "category": "家装›灯具›投光灯",
    "attributes": ["50瓦", "防水"],
    "options": {"颜色": "高亮50W"},
    "price": 42.0,
}


class RewardTests(unittest.TestCase):
    def test_empty_query_is_not_type_evidence(self):
        self.assertFalse(strict_type_match({"asin": "x", "query": "", "category": ""}, {"asin": "y", "query": "", "category": ""}))

    def test_correct_purchase(self):
        result = compute_trajectory_reward(
            {"termination": "purchase", "final_result": {"goal": GOAL, "purchase": CORRECT}},
            {},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["strict_success"], 1.0)

    def test_wrong_price_is_irreversible_failure(self):
        purchase = {**CORRECT, "price": 177.0}
        result = compute_trajectory_reward(
            {"termination": "purchase", "final_result": {"goal": GOAL, "purchase": purchase}},
            {},
        )
        self.assertEqual(result["price_ok"], 0.0)
        self.assertLess(result["score"], -0.8)

    def test_timeout_worse_than_near_miss_purchase(self):
        timeout = compute_trajectory_reward({"termination": "timeout"}, {"goal": GOAL})
        self.assertEqual(timeout["score"], -1.05)

    def test_turn_limit_is_a_timeout(self):
        result = compute_trajectory_reward({"termination": "turn_limit"}, {"goal": GOAL})
        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["score"], -1.05)

    def test_context_limit_is_scored_as_timeout_with_behavior_penalties(self):
        result = compute_trajectory_reward(
            {
                "termination": "context_limit",
                "invalid_count": 1,
                "repeat_count": 1,
            },
            {"goal": GOAL},
        )
        self.assertEqual(result["outcome"], "timeout")
        self.assertAlmostEqual(result["score"], -1.15)

    def test_compute_score_reads_v0_extra_info_trace(self):
        result = compute_score(
            data_source="shopsimulator_grpo",
            solution_str="",
            ground_truth={"goal": GOAL},
            extra_info={"shop_trace": {"termination": "invalid_model"}},
        )
        self.assertEqual(result["outcome"], "invalid_model")

    def test_compute_score_reads_v1_tool_extra_fields_trace(self):
        result = compute_score(
            data_source="shopsimulator_grpo",
            solution_str="",
            ground_truth={"goal": GOAL},
            extra_info={
                "tool_extra_fields": {
                    "shop_trace": {"termination": "turn_limit"},
                }
            },
        )
        self.assertEqual(result["outcome"], "timeout")

    def test_repeat_and_invalid_penalties_are_capped(self):
        result = compute_trajectory_reward(
            {
                "termination": "purchase",
                "repeat_count": 99,
                "invalid_count": 99,
                "final_result": {"goal": GOAL, "purchase": CORRECT},
            },
            {},
        )
        self.assertAlmostEqual(result["score"], 0.7)

    def test_exact_category_is_allowed_as_equivalent_product(self):
        alternative = {**CORRECT, "asin": "alternative"}
        score = evaluate_compliance(
            GOAL,
            alternative,
            selected_options=alternative["options"],
            purchase_price=alternative["price"],
        )
        self.assertTrue(score.strict_success)

    def test_fuzzy_matching_does_not_confuse_numeric_specs(self):
        wrong = {**CORRECT, "attributes": ["500瓦", "防水"], "options": {"颜色": "高亮500W"}}
        score = evaluate_compliance(
            GOAL, wrong, selected_options=wrong["options"], purchase_price=42.0
        )
        self.assertLess(score.attr_score, 1.0)
        self.assertEqual(score.option_score, 0.0)
        self.assertFalse(score.strict_success)


if __name__ == "__main__":
    unittest.main()

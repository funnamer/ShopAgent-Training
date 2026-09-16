import unittest

from web_agent_site.engine.goal import (
    extract_price_upper,
    get_reward,
    get_type_reward,
    resolve_purchase_price,
)


class GoalRewardTest(unittest.TestCase):
    def test_empty_queries_do_not_match(self):
        purchased = {
            "query": "",
            "category": "家装›紧固件",
            "title": "不锈钢螺丝",
        }
        goal = {
            "query": "",
            "category": "运动›健身器械",
            "name": "动感单车刹车片",
        }
        result = get_type_reward(purchased, goal)
        self.assertFalse(result["query_match"])
        self.assertEqual(result["r_type"], 0.5)

    def test_nonempty_equal_queries_still_match(self):
        purchased = {"query": "乳胶枕", "category": "a", "title": "x"}
        goal = {"query": " 乳胶枕 ", "category": "b", "name": "y"}
        self.assertTrue(get_type_reward(purchased, goal)["query_match"])

    def test_price_upper_comes_from_visible_instruction(self):
        self.assertEqual(extract_price_upper("预算在1000元以下。"), 1000.0)
        self.assertEqual(extract_price_upper("价格在40元左右。"), 44.0)
        self.assertEqual(extract_price_upper("预算三十多。"), 40.0)
        self.assertEqual(extract_price_upper("预算大概50多块钱。"), 60.0)
        self.assertEqual(extract_price_upper("价格别超过五百。"), 500.0)
        self.assertEqual(extract_price_upper("价钱在190左右。"), 209.0)
        self.assertEqual(extract_price_upper("价格在10-20之间。"), 20.0)
        self.assertEqual(extract_price_upper("预算4k左右。"), 4400.0)
        self.assertEqual(extract_price_upper("价格在1万到1.1万之间。"), 11000.0)
        self.assertEqual(extract_price_upper("预算4万。"), 40000.0)
        self.assertEqual(extract_price_upper("预算在3百元以内。"), 300.0)
        self.assertIsNone(extract_price_upper("预算4k+。"))
        self.assertIsNone(extract_price_upper("预算在5900元以上。"))
        self.assertEqual(extract_price_upper("想找个位数价格的。"), 9.0)
        self.assertIsNone(extract_price_upper("价格实惠就行。"))

    def test_selected_sku_price_overrides_product_range_fallback(self):
        product = {"option_to_price": {"50w": 42, "100w": 94}}
        self.assertEqual(resolve_purchase_price(product, {"功率": "50w"}, 177), 42.0)
        self.assertEqual(resolve_purchase_price(product, {}, 42), 42.0)

    def test_reward_uses_selected_sku_price_and_reports_it(self):
        purchased = {
            "query": "灯",
            "category": "家装›灯具",
            "title": "50w led灯",
            "Attributes": ["防水"],
            "Title": "50w led灯",
            "BulletPoints": [],
            "Description": "防水",
            "option_to_price": {"50w": 42},
        }
        goal = {
            "query": "灯",
            "category": "家装›灯具",
            "name": "50w led灯",
            "attributes": ["防水"],
            "goal_options": ["50w"],
            "price_upper": 44,
        }
        _, info = get_reward(
            purchased, goal, price=177, options={"功率": "50w"}, verbose=True
        )
        self.assertTrue(info["r_price"])
        self.assertEqual(info["evaluated_price"], 42.0)
        self.assertEqual(info["price_upper"], 44)


if __name__ == "__main__":
    unittest.main()

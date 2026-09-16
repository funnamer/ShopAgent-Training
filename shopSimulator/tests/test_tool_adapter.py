import json
import unittest

from tool_adapter import ShopToolAdapter, assistant_message_to_dict


class FakeShopEnv:
    def __init__(self):
        self.actions = []

    def interact(self, action):
        self.actions.append(action)
        return {
            "instruction": "next page",
            "done": False,
            "over": False,
            "reward": 0,
        }


def tool_message(name, arguments, call_id="call_1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


class ShopToolAdapterTest(unittest.TestCase):
    def test_preserves_provider_reasoning_content_from_sdk_message(self):
        class FakeSdkMessage:
            model_extra = {"reasoning_content": "compare all constraints"}
            reasoning_content = None

            @staticmethod
            def model_dump(exclude_none=True):
                return {"role": "assistant", "content": "", "tool_calls": []}

        self.assertEqual(
            assistant_message_to_dict(FakeSdkMessage())["reasoning_content"],
            "compare all constraints",
        )

    def test_search_is_translated_and_environment_result_is_a_tool_message(self):
        env = FakeShopEnv()
        result = ShopToolAdapter(env).execute_assistant_message(
            tool_message("search", {"keywords": "无线降噪耳机"})
        )

        self.assertEqual(env.actions, ["search[无线降噪耳机]"])
        self.assertEqual(
            set(result.message),
            {"role", "tool_call_id", "content"},
        )
        self.assertEqual(result.message["role"], "tool")
        self.assertEqual(result.message["tool_call_id"], "call_1")
        self.assertEqual(result.message["content"], "next page")
        self.assertEqual(result.output["result"]["instruction"], "next page")

    def test_click_is_translated(self):
        env = FakeShopEnv()
        ShopToolAdapter(env).execute_assistant_message(
            tool_message("click", {"value": "buy now"})
        )
        self.assertEqual(env.actions, ["click[buy now]"])

    def test_click_is_grounded_to_current_clickables(self):
        env = FakeShopEnv()
        adapter = ShopToolAdapter(env)
        adapter.update_observation(
            'page\n搜索功能是否可用: False\n可点击的按钮: ["Buy Now", "< prev"]'
        )

        result = adapter.execute_assistant_message(
            tool_message("click", {"value": "buy now"})
        )

        self.assertTrue(result.output["ok"])
        self.assertEqual(env.actions, ["click[Buy Now]"])

    def test_invalid_click_returns_recoverable_tool_feedback(self):
        env = FakeShopEnv()
        adapter = ShopToolAdapter(env)
        adapter.update_observation(
            'page\n搜索功能是否可用: False\n可点击的按钮: ["back to search", "< prev"]'
        )

        result = adapter.execute_assistant_message(
            tool_message("click", {"value": "description"})
        )

        self.assertFalse(result.output["ok"])
        self.assertTrue(result.output["recoverable"])
        self.assertEqual(result.output["source"], "validation")
        self.assertEqual(env.actions, [])
        self.assertIn("description", result.message["content"])
        self.assertIn("back to search", result.message["content"])

    def test_repeated_action_on_unchanged_state_is_not_sent_twice(self):
        env = FakeShopEnv()
        adapter = ShopToolAdapter(env)
        observation = (
            'page\n搜索功能是否可用: False\n可点击的按钮: ["description"]'
        )
        adapter.update_observation(observation)

        first = adapter.execute_assistant_message(
            tool_message("click", {"value": "description"}, call_id="call_1")
        )
        adapter.update_observation(observation)
        second = adapter.execute_assistant_message(
            tool_message("click", {"value": "description"}, call_id="call_2")
        )

        self.assertTrue(first.output["ok"])
        self.assertFalse(second.output["ok"])
        self.assertTrue(second.output["recoverable"])
        self.assertEqual(env.actions, ["click[description]"])
        self.assertIn("重复动作", second.message["content"])

    def test_ask_shopper_returns_a_tool_message(self):
        env = FakeShopEnv()
        adapter = ShopToolAdapter(env, ask_shopper=lambda question: f"回答：{question}")
        result = adapter.execute_assistant_message(
            tool_message("ask_shopper", {"question": "预算是多少？"})
        )

        self.assertEqual(result.name, "ask_shopper")
        self.assertEqual(result.output["source"], "shopper")
        self.assertIn("预算是多少", result.output["response"])
        self.assertEqual(result.message["role"], "tool")
        self.assertEqual(result.message["content"], "回答：预算是多少？")

    def test_rejects_multiple_tool_calls(self):
        env = FakeShopEnv()
        message = tool_message("click", {"value": "one"})
        message["tool_calls"].append(message["tool_calls"][0])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ShopToolAdapter(env).execute_assistant_message(message)


if __name__ == "__main__":
    unittest.main()

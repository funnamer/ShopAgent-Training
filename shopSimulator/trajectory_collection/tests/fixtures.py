import copy
import json


def valid_trajectory(task_id=1459):
    def call(call_id, name, arguments):
        return {
            "role": "assistant",
            "content": "Thought: teacher-only reasoning",
            "reasoning_content": "private teacher reasoning",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }],
        }

    return {
        "task_id": task_id,
        "reward": 1.0,
        "reward_detail": {
            "r_att": 1.0,
            "r_option": 1.0,
            "r_price": True,
            "r_type": 1.0,
        },
        "goal": {"asin": "123", "goal_options": ["红色"]},
        "purchase": {"asin": "123", "options": {"颜色": "红色"}},
        "conversation": [
            {"role": "system", "content": "购物助手"},
            {
                "role": "user",
                "content": "Instruction: 买红色商品\n搜索功能是否可用: True\n可点击的按钮: []",
            },
            call("call_1", "search", {"keywords": "红色商品"}),
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "搜索功能是否可用: False\n可点击的按钮: [\"123\"]",
            },
            call("call_2", "click", {"value": "123"}),
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": "搜索功能是否可用: False\n可点击的按钮: [\"红色\", \"buy now\"]",
            },
            call("call_3", "click", {"value": "红色"}),
            {
                "role": "tool",
                "tool_call_id": "call_3",
                "content": "搜索功能是否可用: False\n可点击的按钮: [\"红色\", \"buy now\"]",
            },
            call("call_4", "click", {"value": "buy now"}),
            {"role": "tool", "tool_call_id": "call_4", "content": ""},
        ],
    }


def valid_diagnostics(task_id=1459):
    return {
        "task_id": task_id,
        "termination_reason": "purchase",
        "completed_purchase": True,
        "thinking": "enabled",
        "reward": 1.0,
        "reward_detail": {
            "r_att": 1.0,
            "r_option": 1.0,
            "r_price": True,
            "r_type": 1.0,
        },
        "invalid_tool_response_count": 0,
        "tool_errors": [],
        "model_usage": {"completion_tokens": 100, "prompt_tokens": 200},
        "goal_summary": {"asin": "123", "goal_options": ["红色"]},
        "purchase_summary": {"asin": "123", "options": {"颜色": "红色"}},
    }

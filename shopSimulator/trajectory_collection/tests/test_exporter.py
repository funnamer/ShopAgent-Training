import copy
import json
import tempfile
import unittest
from pathlib import Path

from trajectory_collection.exporter import (
    export_reasoning_sample,
    export_sample,
    validate_exported_sample,
    validate_jsonl,
)
from trajectory_collection.tests.fixtures import valid_trajectory


class ExporterTest(unittest.TestCase):
    def test_export_is_action_only_qwen_shape_and_does_not_mutate_raw(self):
        trajectory = valid_trajectory()
        original = copy.deepcopy(trajectory)
        sample = export_sample(trajectory)
        validate_exported_sample(sample)

        self.assertEqual(trajectory, original)
        self.assertEqual(set(sample), {"messages", "tools"})
        assistant_messages = [
            message for message in sample["messages"] if message["role"] == "assistant"
        ]
        self.assertTrue(assistant_messages)
        for message in assistant_messages:
            self.assertEqual(message["content"], "")
            self.assertNotIn("reasoning_content", message)
            self.assertIsInstance(
                message["tool_calls"][0]["function"]["arguments"], dict
            )

    def test_jsonl_can_be_checked_with_native_template_contract(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        sample = export_sample(valid_trajectory())
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
            result = validate_jsonl(path, tokenizer)
        self.assertEqual(result["rows"], 1)
        self.assertTrue(result["native_chat_template_checked"])
        self.assertFalse(tokenizer.kwargs["enable_thinking"])

    def test_terminal_privileged_tool_response_is_cleared(self):
        trajectory = valid_trajectory()
        trajectory["conversation"][-1]["content"] = (
            "Purchased [SEP] asin [SEP] 123 [SEP] Target [SEP] asin "
            "[SEP] Reward Details [SEP] 1.0"
        )
        original = copy.deepcopy(trajectory)

        sample = export_sample(trajectory)

        self.assertEqual(trajectory, original)
        self.assertEqual(sample["messages"][-1]["role"], "tool")
        self.assertEqual(sample["messages"][-1]["content"], "")
        validate_exported_sample(sample)

    def test_reasoning_export_retains_teacher_reasoning_without_mutating_raw(self):
        trajectory = valid_trajectory()
        original = copy.deepcopy(trajectory)

        sample = export_reasoning_sample(trajectory)

        self.assertEqual(trajectory, original)
        assistants = [
            message for message in sample["messages"]
            if message["role"] == "assistant"
        ]
        self.assertTrue(assistants)
        self.assertTrue(all(message["content"] for message in assistants))
        self.assertTrue(all(message["reasoning_content"] for message in assistants))
        self.assertEqual(sample["messages"][-1]["content"], "")
        validate_exported_sample(sample, allow_reasoning=True)

        with self.assertRaisesRegex(ValueError, "action-only"):
            validate_exported_sample(sample)


if __name__ == "__main__":
    unittest.main()

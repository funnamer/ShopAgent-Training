import copy
import unittest

from trajectory_collection.tests.fixtures import valid_diagnostics, valid_trajectory
from trajectory_collection.verifier import verify_trajectory


class VerifierTest(unittest.TestCase):
    def verify(self, trajectory, diagnostics=None):
        if diagnostics is None:
            diagnostics = valid_diagnostics()
            diagnostics["reward"] = trajectory["reward"]
            diagnostics["reward_detail"] = copy.deepcopy(trajectory["reward_detail"])
            diagnostics["goal_summary"] = copy.deepcopy(trajectory["goal"])
            diagnostics["purchase_summary"] = copy.deepcopy(trajectory["purchase"])
        return verify_trajectory(
            trajectory,
            diagnostics,
            1459,
            23421,
        )

    def test_accepts_valid_executed_trajectory(self):
        result = self.verify(valid_trajectory())
        self.assertTrue(result.accepted, result.reasons)
        self.assertEqual(result.metrics["num_tool_calls"], 4)

    def test_rejects_wrong_asin_and_options(self):
        trajectory = valid_trajectory()
        trajectory["purchase"]["asin"] = "wrong"
        trajectory["purchase"]["options"] = {"颜色": "蓝色"}
        result = self.verify(trajectory)
        self.assertIn("asin_mismatch", result.reasons)
        self.assertIn("exact_options_mismatch", result.reasons)

    def test_option_equality_ignores_only_punctuation_and_symbols(self):
        trajectory = valid_trajectory()
        trajectory["goal"]["goal_options"] = ["【?质保十年】亮银/双折"]
        trajectory["purchase"]["options"] = {
            "颜色": "【⭐质保十年】亮银 | 双折"
        }
        result = self.verify(trajectory)
        self.assertTrue(result.accepted, result.reasons)

    def test_click_matching_does_not_use_option_normalization(self):
        trajectory = valid_trajectory()
        trajectory["conversation"][4]["tool_calls"][0]["function"]["arguments"] = (
            '{"value":"1/23"}'
        )
        trajectory["conversation"][3]["content"] = (
            "搜索功能是否可用: False\n可点击的按钮: [\"1|23\"]"
        )
        result = self.verify(trajectory)
        self.assertIn("click_not_available", result.reasons)

    def test_rejects_unavailable_click(self):
        trajectory = valid_trajectory()
        trajectory["conversation"][4]["tool_calls"][0]["function"]["arguments"] = '{"value":"999"}'
        result = self.verify(trajectory)
        self.assertIn("click_not_available", result.reasons)

    def test_rejects_search_when_unavailable(self):
        trajectory = valid_trajectory()
        trajectory["conversation"][4]["tool_calls"][0]["function"] = {
            "name": "search",
            "arguments": '{"keywords":"again"}',
        }
        result = self.verify(trajectory)
        self.assertIn("search_when_unavailable", result.reasons)

    def test_rejects_broken_arguments_and_missing_tool_response(self):
        broken_arguments = valid_trajectory()
        broken_arguments["conversation"][2]["tool_calls"][0]["function"]["arguments"] = "{"
        self.assertIn("invalid_tool_arguments", self.verify(broken_arguments).reasons)

        missing_response = valid_trajectory()
        missing_response["conversation"].pop(3)
        self.assertIn("missing_tool_response", self.verify(missing_response).reasons)

    def test_rejects_hidden_retry(self):
        diagnostics = valid_diagnostics()
        diagnostics["invalid_tool_response_count"] = 1
        result = self.verify(valid_trajectory(), diagnostics)
        self.assertIn("hidden_tool_retry", result.reasons)

    def test_rejects_repeated_state_action_loop(self):
        trajectory = valid_trajectory()
        trajectory["conversation"][6] = copy.deepcopy(trajectory["conversation"][4])
        trajectory["conversation"][6]["tool_calls"][0]["id"] = "call_3"
        trajectory["conversation"][7]["tool_call_id"] = "call_3"
        trajectory["conversation"][5]["content"] = trajectory["conversation"][3]["content"]
        result = self.verify(trajectory)
        self.assertIn("repeated_state_action_loop", result.reasons)

    def test_rejects_privileged_label_leak(self):
        trajectory = valid_trajectory()
        trajectory["conversation"][0]["content"] += ' {"goal_options":["红色"]}'
        result = self.verify(trajectory)
        self.assertIn("privileged_label_leak", result.reasons)

    def test_rejects_task_diagnostic_and_role_sequence_mismatches(self):
        trajectory = valid_trajectory()
        diagnostics = valid_diagnostics()
        diagnostics["task_id"] = 1460
        result = verify_trajectory(
            trajectory,
            diagnostics,
            1459,
            23421,
            expected_task_id=1460,
        )
        self.assertIn("trajectory_task_id_mismatch", result.reasons)
        self.assertIn("diagnostics_task_id_mismatch", result.reasons)

        trajectory = valid_trajectory()
        trajectory["conversation"].insert(4, {"role": "user", "content": "extra"})
        result = self.verify(trajectory)
        self.assertIn("invalid_conversation_role_sequence", result.reasons)
        self.assertIn("not_single_turn_conversation", result.reasons)

    def test_rejects_prompt_and_instruction_hash_mismatches(self):
        result = verify_trajectory(
            valid_trajectory(),
            valid_diagnostics(),
            1459,
            23421,
            expected_system_prompt_sha256="wrong",
            expected_instruction_sha256="wrong",
        )
        self.assertIn("system_prompt_mismatch", result.reasons)
        self.assertIn("instruction_task_mismatch", result.reasons)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import ast
from pathlib import Path
import unittest


AGENT_LOOP_PATH = Path(__file__).resolve().parents[1] / "agent_loop.py"


class AgentLoopContractTests(unittest.TestCase):
    def test_generated_fields_do_not_shadow_dataset_extra_info(self):
        """DataProto.union rejects unequal duplicate non-tensor fields."""
        tree = ast.parse(AGENT_LOOP_PATH.read_text(encoding="utf-8"))
        assigned_extra_field_keys = {
            node.slice.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "extra_fields"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and isinstance(getattr(node, "ctx", None), ast.Store)
        }

        self.assertIn("shop_trace", assigned_extra_field_keys)
        self.assertNotIn("extra_info", assigned_extra_field_keys)


if __name__ == "__main__":
    unittest.main()

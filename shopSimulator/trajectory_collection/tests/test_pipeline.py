import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from trajectory_collection.common import sha256_file
from trajectory_collection.pipeline import _is_complete_pair, collect


class FakeSingleAgent:
    calls = []

    @staticmethod
    def load_env_file():
        return None

    @classmethod
    def run_tasks_multithreaded(cls, task_ids, config, max_workers):
        run_name = config["agent_config"]["run_name"]
        output = Path(config["agent_config"]["output_path"]) / run_name
        (output / "diagnostics").mkdir(parents=True, exist_ok=True)
        cls.calls.append((list(task_ids), run_name, max_workers))
        for task_id in task_ids:
            (output / f"{task_id}.json").write_text(
                json.dumps({"task_id": task_id}), encoding="utf-8"
            )
            (output / "diagnostics" / f"{task_id}.json").write_text(
                json.dumps({"task_id": task_id}), encoding="utf-8"
            )


class PipelineTest(unittest.TestCase):
    def setUp(self):
        FakeSingleAgent.calls = []

    def _fixture(self, root):
        teacher = root / "teacher.yaml"
        teacher.write_text(yaml.safe_dump({
            "env_config": {"if_persona": False},
            "agent_config": {
                "model_name": "deepseek-chat",
                "source": "deepseek",
                "thinking": "enabled",
                "system_prompt": "unchanged prompt",
            },
        }), encoding="utf-8")
        manifest = root / "ids.json"
        manifest.write_text(json.dumps([1459, 1460]), encoding="utf-8")
        config = {
            "collection_name": "test",
            "teacher_config": str(teacher),
            "output_root": str(root / "outputs"),
            "train_start": 1459,
            "train_end": 23421,
            "rollouts_per_task": 4,
        }
        metadata = {
            1459: {"instruction_sha256": "a"},
            1460: {"instruction_sha256": "b"},
        }
        return config, manifest, metadata

    def test_rollouts_are_isolated_and_resume_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, metadata = self._fixture(Path(directory))
            with patch("trajectory_collection.pipeline._import_single_eval_agent", return_value=FakeSingleAgent), patch("trajectory_collection.pipeline._task_metadata", return_value=metadata):
                first = collect(config, manifest, 4, 8)
                second = collect(config, manifest, 4, 8)
            self.assertEqual(len(FakeSingleAgent.calls), 4)
            self.assertEqual([call[1] for call in FakeSingleAgent.calls], [f"rollout-{i:02d}" for i in range(4)])
            self.assertTrue(all(run["collected_now"] == 2 for run in first["runs"]))
            self.assertTrue(all(run["collected_now"] == 0 for run in second["runs"]))
            self.assertEqual(second["manifest_sha256"], sha256_file(manifest))

    def test_incomplete_pair_is_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, metadata = self._fixture(Path(directory))
            with patch("trajectory_collection.pipeline._import_single_eval_agent", return_value=FakeSingleAgent), patch("trajectory_collection.pipeline._task_metadata", return_value=metadata):
                collect(config, manifest, 4, 8)
                diagnostics = Path(config["output_root"]) / "test/raw/rollout-00/diagnostics/1459.json"
                diagnostics.unlink()
                self.assertFalse(_is_complete_pair(diagnostics.parents[1], 1459))
                collect(config, manifest, 4, 8)
            self.assertEqual(FakeSingleAgent.calls[-1][:2], ([1459], "rollout-00"))
            self.assertEqual(len(FakeSingleAgent.calls), 5)

if __name__ == "__main__":
    unittest.main()

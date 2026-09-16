import json
import random
import tempfile
import unittest
from pathlib import Path

from trajectory_collection.sampler import (
    choose_pilot,
    deduplicate_and_refill,
    difficulty_for,
    load_eligible_tasks,
    proportional_sample,
    stratified_sample,
)


class SamplerTest(unittest.TestCase):
    def test_difficulty_boundaries(self):
        self.assertEqual(difficulty_for(1), "simple")
        self.assertEqual(difficulty_for(3), "simple")
        self.assertEqual(difficulty_for(4), "medium")
        self.assertEqual(difficulty_for(6), "medium")
        self.assertEqual(difficulty_for(7), "hard")

    def test_loader_is_train_only_and_excludes_eval_instruction_overlap(self):
        def item(tag, instruction, attrs=1, domain="A", options=1):
            return {
                "tag": tag,
                "domain_zh": domain,
                "instructions": [{
                    "instruction": instruction,
                    "attributes": list(range(attrs)),
                    "instruction_options": list(range(options)),
                }],
            }

        data = [
            item("eval", "相同 指令"),
            item("eval", "评测二"),
            item("train", "相同指令"),
            item("train", "训练一"),
            item("train", "训练二"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            eligible, audit = load_eligible_tasks(path, 2, 5)

        self.assertEqual([record["task_id"] for record in eligible], [3, 4])
        self.assertEqual(audit["excluded_eval_instruction_overlap_ids"], [2])

    def test_stratified_sample_is_reproducible_and_respects_quotas(self):
        records = []
        for difficulty in ("simple", "medium", "hard"):
            for domain in ("A", "B"):
                for option_bucket in ("single", "multi"):
                    for _ in range(4):
                        records.append({
                            "task_id": len(records),
                            "difficulty": difficulty,
                            "domain": domain,
                            "option_bucket": option_bucket,
                        })
        quotas = {"simple": 8, "medium": 8, "hard": 8}
        first = stratified_sample(records, quotas, random.Random(42))
        second = stratified_sample(records, quotas, random.Random(42))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        for difficulty in quotas:
            self.assertEqual(
                sum(record["difficulty"] == difficulty for record in first), 8
            )

    def test_pilot_covers_domains_and_adds_hard_multi(self):
        records = []
        for domain in ("A", "B", "C"):
            for index in range(3):
                records.append({
                    "task_id": len(records),
                    "domain": domain,
                    "difficulty": "hard" if index == 2 else "simple",
                    "option_bucket": "multi" if index == 2 else "single",
                })
        pilot = choose_pilot(records, 4, 43)
        self.assertEqual({r["domain"] for r in pilot}, {"A", "B", "C"})
        self.assertTrue(
            any(r["difficulty"] == "hard" and r["option_bucket"] == "multi" for r in pilot)
        )

    def test_duplicate_instructions_are_refilled_in_the_same_stratum(self):
        selected = [
            {"task_id": 1, "difficulty": "hard", "domain": "A", "option_bucket": "multi", "instruction_sha256": "same"},
            {"task_id": 2, "difficulty": "hard", "domain": "A", "option_bucket": "multi", "instruction_sha256": "same"},
        ]
        pool = [
            *selected,
            {"task_id": 3, "difficulty": "hard", "domain": "A", "option_bucket": "multi", "instruction_sha256": "fresh"},
        ]
        result, audit = deduplicate_and_refill(selected, pool)
        self.assertEqual([record["task_id"] for record in result], [1, 3])
        self.assertEqual(audit["removed_duplicate_task_ids"], [2])
        self.assertEqual(audit["remaining_internal_duplicate_count"], 0)

    def test_proportional_sample_uses_configured_difficulty_ratio(self):
        records = [
            {"task_id": task_id, "difficulty": difficulty}
            for difficulty, count in (("simple", 60), ("medium", 100), ("hard", 40))
            for task_id in range(
                {"simple": 0, "medium": 60, "hard": 160}[difficulty],
                {"simple": 0, "medium": 60, "hard": 160}[difficulty] + count,
            )
        ]
        first, quotas = proportional_sample(
            records,
            50,
            {"simple": 600, "medium": 1000, "hard": 400},
            random.Random(92),
        )
        second, _ = proportional_sample(
            records,
            50,
            {"simple": 600, "medium": 1000, "hard": 400},
            random.Random(92),
        )

        self.assertEqual(quotas, {"simple": 15, "medium": 25, "hard": 10})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)


if __name__ == "__main__":
    unittest.main()

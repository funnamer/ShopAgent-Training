import unittest

from trajectory_collection.pipeline import _split_final_records


class FinalSplitTest(unittest.TestCase):
    def test_final_split_is_deterministic_disjoint_and_quota_exact(self):
        accepted = {"simple": 450, "medium": 750, "hard": 300}
        records = []
        for difficulty, count in accepted.items():
            for index in range(count):
                task_id = len(records) + 1459
                records.append(
                    {
                        "task_id": task_id,
                        "difficulty": difficulty,
                        "domain": f"domain-{index % 3}",
                        "option_bucket": "multi" if index % 4 == 0 else "single",
                        "instruction_sha256": f"hash-{task_id}",
                    }
                )
        config = {
            "seed": 42,
            "sft_split_sizes": {"train": 1000, "validation": 100, "reserve": 400},
            "sft_split_difficulty_quotas": {
                "train": {"simple": 300, "medium": 500, "hard": 200},
                "validation": {"simple": 30, "medium": 50, "hard": 20},
                "reserve": {"simple": 120, "medium": 200, "hard": 80},
            },
        }
        first = _split_final_records(records, config)
        second = _split_final_records(records, config)
        self.assertEqual(first, second)
        self.assertEqual({key: len(value) for key, value in first.items()}, config["sft_split_sizes"])
        id_sets = [{record["task_id"] for record in values} for values in first.values()]
        self.assertEqual(sum(map(len, id_sets)), len(set().union(*id_sets)))
        for split, quotas in config["sft_split_difficulty_quotas"].items():
            for difficulty, count in quotas.items():
                self.assertEqual(
                    sum(record["difficulty"] == difficulty for record in first[split]),
                    count,
                )

    def test_duplicate_instruction_is_rejected_before_split(self):
        records = [
            {"task_id": 1, "instruction_sha256": "same"},
            {"task_id": 2, "instruction_sha256": "same"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _split_final_records(records, {"seed": 42})


if __name__ == "__main__":
    unittest.main()

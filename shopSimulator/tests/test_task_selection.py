import unittest

from task_selection import select_task_ids, task_selection_metadata


class TaskSelectionTest(unittest.TestCase):
    def test_default_preserves_original_sequential_behavior(self):
        self.assertEqual(select_task_ids({"task_nums": 5}), [0, 1, 2, 3, 4])

    def test_sequential_sample_takes_prefix(self):
        config = {
            "task_nums": 10,
            "task_selection": {"mode": "sequential", "sample_size": 4},
        }
        self.assertEqual(select_task_ids(config), [0, 1, 2, 3])

    def test_random_sample_is_reproducible(self):
        config = {
            "task_nums": 100,
            "task_selection": {"mode": "random", "sample_size": 8, "seed": 42},
        }
        first = select_task_ids(config)
        second = select_task_ids(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(set(first)), 8)

        changed_seed = {
            **config,
            "task_selection": {"mode": "random", "sample_size": 8, "seed": 43},
        }
        self.assertNotEqual(first, select_task_ids(changed_seed))

    def test_random_selection_metadata_records_seed_and_ids(self):
        config = {
            "task_nums": 20,
            "task_selection": {"mode": "random", "sample_size": 3, "seed": 7},
        }
        task_ids = select_task_ids(config)
        self.assertEqual(
            task_selection_metadata(config, task_ids),
            {
                "mode": "random",
                "seed": 7,
                "task_pool_size": 20,
                "sample_size": 3,
                "selected_task_ids": task_ids,
            },
        )

    def test_rejects_invalid_selection(self):
        invalid_configs = [
            {"task_nums": -1},
            {"task_nums": 10, "task_selection": {"mode": "unknown"}},
            {
                "task_nums": 10,
                "task_selection": {"mode": "random", "sample_size": 11},
            },
            {
                "task_nums": 10,
                "task_selection": {"mode": "random", "seed": "42"},
            },
        ]
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    select_task_ids(config)


if __name__ == "__main__":
    unittest.main()

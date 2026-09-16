import copy
import unittest

from trajectory_collection.common import validate_collection_config


VALID_CONFIG = {
    "candidate_size": 20,
    "online_dev_size": 4,
    "difficulty_quotas": {"simple": 6, "medium": 10, "hard": 4},
    "online_dev_difficulty_quotas": {"simple": 1, "medium": 2, "hard": 1},
    "accepted_difficulty_quotas": {"simple": 4, "medium": 5, "hard": 1},
    "sft_split_sizes": {"train": 8, "validation": 2},
    "sft_split_difficulty_quotas": {
        "train": {"simple": 3, "medium": 4, "hard": 1},
        "validation": {"simple": 1, "medium": 1, "hard": 0},
    },
}


class ConfigTest(unittest.TestCase):
    def test_accepts_consistent_sizes_and_quotas(self):
        validate_collection_config(copy.deepcopy(VALID_CONFIG))

    def test_rejects_candidate_size_mismatch(self):
        config = copy.deepcopy(VALID_CONFIG)
        config["candidate_size"] = 21
        with self.assertRaisesRegex(ValueError, "candidate_size"):
            validate_collection_config(config)

    def test_rejects_split_difficulty_mismatch(self):
        config = copy.deepcopy(VALID_CONFIG)
        config["sft_split_difficulty_quotas"]["train"]["hard"] = 0
        config["sft_split_difficulty_quotas"]["train"]["simple"] = 4
        with self.assertRaisesRegex(ValueError, "split quota sum"):
            validate_collection_config(config)


if __name__ == "__main__":
    unittest.main()

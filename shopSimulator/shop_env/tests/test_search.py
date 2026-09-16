import tempfile
import unittest
from pathlib import Path

from web_agent_site.engine.search import (
    MultiFieldBM25Searcher,
    SearchIndexError,
    build_index,
    normalize_query,
    product_data_fingerprint,
    product_fields,
    search_tokens,
)


PRODUCTS = [
    {
        "asin": "000000000003",
        "title": "迷你单门冰箱",
        "shop_name": "小熊",
        "category": "家电›冰箱",
        "attribute": ["宿舍", "静音"],
        "customization_options": {"容量": [{"value": "45升"}]},
    },
    {
        "asin": "000000000001",
        "title": "宿舍小冰箱",
        "shop_name": "海尔",
        "category": "家电›冰箱",
        "attribute": ["节能"],
        "customization_options": {"颜色": [{"value": "白色"}]},
    },
    {
        "asin": "000000000002",
        "title": "开放式耳机 X100",
        "category": "数码›耳机",
        "attribute": ["低夹耳压力"],
    },
]


class SearchTest(unittest.TestCase):
    def test_normalization_and_chinese_tokens_are_deterministic(self):
        self.assertEqual(normalize_query(" 迷你，冰箱 100 CNY "), "迷你 冰箱 100元")
        self.assertEqual(search_tokens("小冰箱"), ("小冰", "冰箱"))

    def test_product_fields_exclude_private_goal_and_reward(self):
        fields = product_fields(
            {
                **PRODUCTS[0],
                "goal": {"asin": "secret"},
                "reward": 1.0,
                "instructions": [{"instruction": "secret target"}],
            }
        )
        combined = " ".join(fields.values())
        self.assertNotIn("secret", combined)

    def test_weighted_search_and_tie_break_are_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "products.sqlite3"
            fingerprint = product_data_fingerprint(PRODUCTS)
            build_index(
                PRODUCTS,
                index_path,
                product_data_sha256=fingerprint,
            )
            searcher = MultiFieldBM25Searcher(
                index_path,
                expected_product_sha256=fingerprint,
            )
            try:
                first = searcher.search("海尔 冰箱")
                second = searcher.search("海尔 冰箱")
                self.assertEqual(first, second)
                self.assertEqual(first[0].asin, "000000000001")
                self.assertEqual(searcher.search(""), [])
            finally:
                searcher.close()

    def test_index_rejects_a_different_product_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "products.sqlite3"
            build_index(PRODUCTS, index_path, product_data_sha256="original")
            with self.assertRaises(SearchIndexError):
                MultiFieldBM25Searcher(
                    index_path,
                    expected_product_sha256="different",
                )


if __name__ == "__main__":
    unittest.main()

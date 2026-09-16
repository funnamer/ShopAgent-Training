from __future__ import annotations

import unittest

from rl.budget import extract_price_upper


class BudgetTests(unittest.TestCase):
    def test_approximate_budget(self):
        self.assertEqual(extract_price_upper("价格在40元左右"), 44.0)

    def test_exact_upper(self):
        self.assertEqual(extract_price_upper("预算100元以内"), 100.0)

    def test_lower_bound_has_no_upper(self):
        self.assertIsNone(extract_price_upper("预算4k+"))


if __name__ == "__main__":
    unittest.main()

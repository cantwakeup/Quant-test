import unittest

import pandas as pd

from src.data_cleaning import align_fundamentals_by_announcement


class FinancialAlignmentTests(unittest.TestCase):
    def test_report_period_does_not_leak_before_ann_date(self):
        prices = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=6)})
        fin = pd.DataFrame({"report_date": ["2023-12-31"], "ann_date": ["2024-01-04"], "revenue": [100.0]})
        aligned = align_fundamentals_by_announcement(prices, fin)
        self.assertTrue(aligned.loc[:2, "revenue"].isna().all())
        self.assertEqual(aligned.loc[3, "revenue"], 100.0)


if __name__ == "__main__":
    unittest.main()

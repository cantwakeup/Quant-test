import unittest

import numpy as np
import pandas as pd

from src.data_cleaning import align_fundamentals_by_announcement
from src.factor_analysis import feature_columns
from src.feature_selection import numeric_feature_columns
from src.label_builder import build_labels


class NoLeakageTests(unittest.TestCase):
    def test_labels_are_shifted_forward(self):
        prices = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=6, freq="D"),
                "open": [1, 2, 3, 4, 5, 6],
                "high": [1, 2, 3, 4, 5, 6],
                "low": [1, 2, 3, 4, 5, 6],
                "close": [1, 2, 4, 8, 16, 32],
                "volume": 1,
                "amount": 1,
                "turnover": 1,
                "pct_change": 0,
            }
        )
        labels = build_labels(prices, horizons=[1, 3])
        self.assertAlmostEqual(labels.loc[0, "y_ret_1d"], np.log(2 / 1))
        self.assertAlmostEqual(labels.loc[0, "y_ret_3d"], np.log(8 / 1))
        self.assertTrue(pd.isna(labels.loc[5, "y_ret_1d"]))

    def test_fundamentals_use_announcement_date(self):
        prices = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5, freq="D")})
        fundamentals = pd.DataFrame(
            {
                "ann_date": [pd.Timestamp("2024-01-03")],
                "report_date": [pd.Timestamp("2023-12-31")],
                "revenue": [100.0],
            }
        )
        aligned = align_fundamentals_by_announcement(prices, fundamentals)
        self.assertTrue(pd.isna(aligned.loc[1, "revenue"]))
        self.assertEqual(aligned.loc[2, "revenue"], 100.0)
        self.assertTrue((aligned.dropna(subset=["ann_date"])["ann_date"] <= aligned.dropna(subset=["ann_date"])["date"]).all())

    def test_factor_columns_exclude_labels(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "ret_5d": [0.1, 0.2, 0.3],
                "y_ret_5d": [0.4, 0.5, 0.6],
                "future_mdd_5d": [-0.1, -0.2, -0.3],
            }
        )
        self.assertEqual(feature_columns(df), ["ret_5d"])
        self.assertEqual(numeric_feature_columns(df), ["ret_5d"])

    def test_new_trade_labels_are_forward_only(self):
        prices = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=8, freq="D"),
                "open": [10, 10, 11, 12, 11, 13, 14, 15],
                "high": [10, 11, 12, 13, 12, 14, 15, 16],
                "low": [9, 9, 10, 10, 10, 12, 13, 14],
                "close": [10, 11, 12, 11, 13, 14, 15, 16],
                "volume": 1,
                "amount": 1,
                "turnover": 1,
                "pct_change": 0,
            }
        )
        labels = build_labels(prices, horizons=[5])
        self.assertIn("future_mfe_5d", labels.columns)
        self.assertIn("future_mae_5d", labels.columns)
        self.assertIn("y_good_trade_5d", labels.columns)
        self.assertTrue(pd.isna(labels.loc[7, "future_mfe_5d"]))


if __name__ == "__main__":
    unittest.main()

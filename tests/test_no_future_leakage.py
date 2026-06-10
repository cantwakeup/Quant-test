import unittest

import pandas as pd

from src.feature_engineering import build_feature_table
from src.feature_selection import numeric_feature_columns, select_features_train_only


class NoFutureLeakageTests(unittest.TestCase):
    def _prices(self):
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "open": range(100, 200),
                "high": range(101, 201),
                "low": range(99, 199),
                "close": range(100, 200),
                "volume": range(1000, 1100),
                "amount": range(10000, 10100),
                "turnover": 1.0,
                "pct_change": 0.1,
                "limit_up": False,
                "limit_down": False,
                "is_suspended": False,
            }
        )

    def test_future_price_change_does_not_change_historical_features(self):
        prices = self._prices()
        base = build_feature_table(prices)
        changed = prices.copy()
        changed.loc[80:, "close"] = changed.loc[80:, "close"] * 5
        changed_features = build_feature_table(changed)
        cols = [c for c in base.columns if c != "date"]
        pd.testing.assert_frame_equal(base.loc[:60, cols], changed_features.loc[:60, cols], check_dtype=False)

    def test_targets_and_future_columns_excluded_from_selection(self):
        df = pd.DataFrame(
            {
                "feature_a": [1, 2, 3, 4],
                "y_ret_5d": [4, 3, 2, 1],
                "future_mfe_5d": [1, 1, 1, 1],
            }
        )
        selected, _ = select_features_train_only(df, pd.Series([0.1, 0.2, 0.3, 0.4]), min_non_null=2)
        self.assertEqual(numeric_feature_columns(df), ["feature_a"])
        self.assertEqual(selected, ["feature_a"])


if __name__ == "__main__":
    unittest.main()

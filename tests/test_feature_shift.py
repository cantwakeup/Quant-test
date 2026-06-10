import unittest

import pandas as pd

from src.data_cleaning import clean_ohlcv
from src.feature_engineering import build_feature_table


class FeatureShiftTests(unittest.TestCase):
    def test_features_do_not_change_when_future_price_changes(self):
        raw = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=80, freq="D"),
                "open": range(80, 160),
                "high": range(81, 161),
                "low": range(79, 159),
                "close": range(80, 160),
                "volume": range(1000, 1080),
                "amount": range(10000, 10080),
                "turnover": 1.0,
                "pct_change": 0.1,
            }
        )
        prices = clean_ohlcv(raw)
        baseline = build_feature_table(prices)
        changed = prices.copy()
        changed.loc[60:, "close"] = changed.loc[60:, "close"] * 10
        changed_features = build_feature_table(changed)
        cols = [c for c in baseline.columns if c != "date"]
        pd.testing.assert_frame_equal(
            baseline.loc[:50, cols].reset_index(drop=True),
            changed_features.loc[:50, cols].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()

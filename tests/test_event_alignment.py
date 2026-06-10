import unittest

import pandas as pd

from src.feature_engineering import build_event_features


class EventAlignmentTests(unittest.TestCase):
    def test_event_cooling_starts_after_event_date(self):
        dates = pd.date_range("2024-01-01", periods=7)
        events = pd.DataFrame({"event_date": [pd.Timestamp("2024-01-04")], "event_type": ["earnings_forecast"]})
        features = build_event_features(pd.Series(dates), events)
        self.assertEqual(features.loc[2, "post_earnings_forecast_5d"], 0.0)
        self.assertEqual(features.loc[3, "post_earnings_forecast_5d"], 1.0)
        self.assertEqual(features.loc[3, "event_cooling_any_risk"], 1.0)


if __name__ == "__main__":
    unittest.main()

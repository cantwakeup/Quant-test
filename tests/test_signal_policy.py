import unittest

import pandas as pd

from src.portfolio_policy import enrich_policy_fields
from src.signal_generator import generate_signals


class SignalPolicyTests(unittest.TestCase):
    def test_high_risk_forces_risk_off_zero_position(self):
        prices = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2), "close": [10, 10], "limit_up": False, "limit_down": False, "is_suspended": False})
        preds = pd.DataFrame({"date": prices["date"], "pred_ret_5d": [0.1, 0.1], "pred_ret_20d": [0.1, 0.1], "prob_up_5d": [0.9, 0.9], "prob_up_20d": [0.9, 0.9]})
        feats = pd.DataFrame({"date": prices["date"], "volatility_20d": [1, 1], "max_drawdown_20d": [-1, -1], "ma_gap_20d": [0.1, 0.1], "trend_consistency_20d": [1, 1], "volume_ratio_20d": [1, 1]})
        sig = generate_signals(prices, preds, feats, {"risk_off_vol_20d": 0.04, "risk_off_drawdown_20d": -0.1})
        self.assertEqual(sig.loc[0, "signal_label"], "risk_off")
        self.assertEqual(sig.loc[0, "target_position"], 0.0)

    def test_missing_external_data_requires_manual_review(self):
        signals = pd.DataFrame({"date": ["2024-01-01"], "target_position": [1.0], "risk_score": [0.2], "reason": ["test"], "stop_loss_reference": [9], "prob_up_20d": [0.6], "signal_score": [0.5]})
        out = enrich_policy_fields(signals, data_quality_flag="missing_external_data")
        self.assertTrue(out.loc[0, "manual_review_required"])


if __name__ == "__main__":
    unittest.main()

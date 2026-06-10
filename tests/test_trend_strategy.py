import unittest

import pandas as pd

from src.backtest import run_vector_backtest
from src.strategy_rules import ma_cross_signal, momentum_signal


class TrendStrategyTests(unittest.TestCase):
    def _prices(self):
        close = list(range(1, 151))
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=150),
                "open": close,
                "high": [x + 1 for x in close],
                "low": [x - 1 for x in close],
                "close": close,
                "volume": 1000,
                "amount": 10000,
                "turnover": 1,
                "pct_change": 0,
                "limit_up": False,
                "limit_down": False,
                "is_suspended": False,
            }
        )

    def test_ma_and_momentum_signals_use_past_data(self):
        prices = self._prices()
        ma = ma_cross_signal(prices, 60, 120)
        mom = momentum_signal(prices, 20)
        changed = prices.copy()
        changed.loc[140:, "close"] = 1
        ma_changed = ma_cross_signal(changed, 60, 120)
        mom_changed = momentum_signal(changed, 20)
        pd.testing.assert_series_equal(ma.loc[:130, "target_position"], ma_changed.loc[:130, "target_position"])
        pd.testing.assert_series_equal(mom.loc[:130, "target_position"], mom_changed.loc[:130, "target_position"])

    def test_next_open_execution_uses_prior_signal(self):
        prices = self._prices().head(5)
        sig = pd.DataFrame({"date": prices["date"], "target_position": [0, 1, 1, 0, 0]})
        bt, _ = run_vector_backtest(prices, sig, config={"commission": 0, "slippage": 0, "stamp_tax": 0})
        self.assertEqual(bt.loc[0, "position"], 0)
        self.assertEqual(bt.loc[1, "position"], 0)
        self.assertEqual(bt.loc[2, "position"], 1)


if __name__ == "__main__":
    unittest.main()

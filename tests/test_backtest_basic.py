import unittest

import pandas as pd

from src.backtest import run_vector_backtest


class BacktestBasicTests(unittest.TestCase):
    def test_long_signal_on_rising_prices_makes_money(self):
        prices = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "open": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                "close": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                "limit_up": False,
                "limit_down": False,
                "is_suspended": False,
            }
        )
        signals = pd.DataFrame({"date": prices["date"], "target_position": 1.0})
        bt, metrics = run_vector_backtest(prices, signals, config={"commission": 0, "stamp_tax": 0, "slippage": 0})
        self.assertGreater(bt["strategy_equity"].iloc[-1], 1.0)
        strategy = metrics.loc[metrics["portfolio"] == "strategy"].iloc[0]
        self.assertGreater(strategy["total_return"], 0)


if __name__ == "__main__":
    unittest.main()

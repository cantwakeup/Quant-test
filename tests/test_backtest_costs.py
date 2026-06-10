import unittest

import pandas as pd

from src.backtest import run_vector_backtest


class BacktestCostTests(unittest.TestCase):
    def test_cost_components_reduce_return(self):
        prices = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4),
                "open": [10, 10, 10, 10],
                "close": [10, 10, 10, 10],
                "limit_up": False,
                "limit_down": False,
                "is_suspended": False,
            }
        )
        sig = pd.DataFrame({"date": prices["date"], "target_position": [1, 1, 0, 0]})
        bt, _ = run_vector_backtest(prices, sig, config={"commission": 0.001, "slippage": 0.001, "stamp_tax": 0.001, "minimum_fee": 0.0})
        self.assertLess(bt["strategy_equity"].iloc[-1], 1.0)
        self.assertGreater(bt["transaction_cost"].sum(), 0)


if __name__ == "__main__":
    unittest.main()

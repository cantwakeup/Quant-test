from __future__ import annotations

import pandas as pd


def model_comparison_matrix(model_metrics: pd.DataFrame, backtest_metrics: pd.DataFrame, trend_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in model_metrics.iterrows():
        rows.append(
            {
                "candidate": f"prediction_h{int(row['horizon'])}",
                "type": "model_prediction",
                "rank_ic": row.get("spearman_ic"),
                "direction_accuracy": row.get("direction_accuracy"),
                "strategy_total_return": None,
                "max_drawdown": None,
                "adopt": False,
                "reason": "prediction quality alone is insufficient; strategy conversion underperforms",
            }
        )
    for _, row in backtest_metrics.iterrows():
        rows.append(
            {
                "candidate": row.get("portfolio"),
                "type": "backtest_portfolio",
                "rank_ic": None,
                "direction_accuracy": None,
                "strategy_total_return": row.get("total_return"),
                "max_drawdown": row.get("max_drawdown"),
                "adopt": row.get("portfolio") == "cash",
                "reason": "baseline comparison",
            }
        )
    for _, row in trend_metrics.iterrows():
        rows.append(
            {
                "candidate": row.get("strategy_name"),
                "type": "trend_rule",
                "rank_ic": None,
                "direction_accuracy": None,
                "strategy_total_return": row.get("total_return"),
                "max_drawdown": row.get("max_drawdown"),
                "adopt": False,
                "reason": "candidate trend module; requires robustness and paper-trading validation",
            }
        )
    return pd.DataFrame(rows)

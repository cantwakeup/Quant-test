from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .backtest import run_vector_backtest


def random_signal_same_trade_count(prices: pd.DataFrame, base_signal: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    position = base_signal["target_position"].fillna(0.0).to_numpy()
    active_days = int((position > 0).sum())
    out = np.zeros(len(position), dtype=float)
    if active_days > 0:
        idx = rng.choice(np.arange(len(position)), size=min(active_days, len(position)), replace=False)
        out[idx] = 1.0
    return pd.DataFrame({"date": base_signal["date"], "target_position": out})


def random_signal_same_holding_blocks(prices: pd.DataFrame, base_signal: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos = base_signal["target_position"].fillna(0.0).astype(float)
    groups = pos.ne(pos.shift()).cumsum()
    lengths = pos.groupby(groups).agg(["first", "size"])
    active_lengths = lengths.loc[lengths["first"] > 0, "size"].astype(int).tolist()
    out = np.zeros(len(pos), dtype=float)
    for length in active_lengths:
        if length <= 0 or length >= len(out):
            continue
        start = int(rng.integers(0, len(out) - length))
        out[start : start + length] = 1.0
    return pd.DataFrame({"date": base_signal["date"], "target_position": out})


def null_signal_test(
    prices: pd.DataFrame,
    base_signal: pd.DataFrame,
    base_total_return: float,
    config: Dict[str, object],
    n_iter: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for i in range(n_iter):
        for method, builder in (
            ("same_trade_count", random_signal_same_trade_count),
            ("same_holding_blocks", random_signal_same_holding_blocks),
        ):
            sig = builder(prices, base_signal, seed=seed + i)
            _, metrics = run_vector_backtest(prices, sig, config=config)
            strat = metrics[metrics["portfolio"] == "strategy"].iloc[0]
            rows.append({"method": method, "iteration": i, "total_return": strat["total_return"], "beats_base": strat["total_return"] >= base_total_return})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["base_total_return"] = base_total_return
    return result

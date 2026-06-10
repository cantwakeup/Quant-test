from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _future_max_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    values = close.astype(float).to_numpy()
    out = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        path = values[i + 1 : i + horizon + 1]
        if len(path) < horizon or not np.isfinite(values[i]):
            continue
        wealth = np.concatenate([[values[i]], path])
        running_max = np.maximum.accumulate(wealth)
        out[i] = np.min(wealth / running_max - 1.0)
    return pd.Series(out, index=close.index)


def _future_volatility(close: pd.Series, horizon: int) -> pd.Series:
    returns = np.log(close / close.shift(1))
    out = np.full(len(close), np.nan, dtype=float)
    for i in range(len(close)):
        sample = returns.iloc[i + 1 : i + horizon + 1].dropna()
        if len(sample) == horizon:
            out[i] = float(sample.std(ddof=0))
    return pd.Series(out, index=close.index)


def build_labels(
    prices: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
    benchmark: Optional[pd.DataFrame] = None,
    crash_threshold_5d: float = -0.08,
    crash_threshold_20d: float = -0.15,
) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("prices is empty.")
    labels = pd.DataFrame({"date": prices["date"].values})
    close = prices["close"].astype(float)

    benchmark_ret = {}
    if benchmark is not None and not benchmark.empty:
        bench = benchmark[["date", "close"]].rename(columns={"close": "benchmark_close"})
        tmp = prices[["date"]].merge(bench, on="date", how="left")
        bclose = tmp["benchmark_close"].ffill()
        for horizon in horizons:
            benchmark_ret[horizon] = np.log(bclose.shift(-horizon) / bclose)

    for horizon in horizons:
        future_ret = np.log(close.shift(-horizon) / close)
        labels[f"y_ret_{horizon}d"] = future_ret
        labels[f"y_up_{horizon}d"] = (future_ret > 0).astype(float)
        labels.loc[future_ret.isna(), f"y_up_{horizon}d"] = np.nan
        labels[f"future_mdd_{horizon}d"] = _future_max_drawdown(close, horizon)
        labels[f"future_vol_{horizon}d"] = _future_volatility(close, horizon)
        if horizon in benchmark_ret:
            labels[f"y_outperform_index_{horizon}d"] = (future_ret > benchmark_ret[horizon]).astype(float)
            labels.loc[future_ret.isna() | benchmark_ret[horizon].isna(), f"y_outperform_index_{horizon}d"] = np.nan

    if "future_mdd_5d" in labels.columns:
        labels["y_crash_5d"] = (labels["future_mdd_5d"] <= crash_threshold_5d).astype(float)
        labels.loc[labels["future_mdd_5d"].isna(), "y_crash_5d"] = np.nan
    if "future_mdd_20d" in labels.columns:
        labels["y_crash_20d"] = (labels["future_mdd_20d"] <= crash_threshold_20d).astype(float)
        labels.loc[labels["future_mdd_20d"].isna(), "y_crash_20d"] = np.nan
    return labels

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


def _future_mfe_mae(close: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    values = close.astype(float).to_numpy()
    mfe = np.full(len(values), np.nan, dtype=float)
    mae = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        path = values[i + 1 : i + horizon + 1]
        if len(path) < horizon or not np.isfinite(values[i]) or values[i] == 0:
            continue
        future_returns = path / values[i] - 1.0
        mfe[i] = float(np.nanmax(future_returns))
        mae[i] = float(np.nanmin(future_returns))
    return pd.Series(mfe, index=close.index), pd.Series(mae, index=close.index)


def build_labels(
    prices: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
    benchmark: Optional[pd.DataFrame] = None,
    estimated_round_trip_cost: float = 0.0021,
    good_trade_return_5d: float = 0.02,
    good_trade_return_20d: float = 0.05,
    good_trade_mae_5d: float = -0.05,
    good_trade_mae_20d: float = -0.10,
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
        labels[f"y_up_after_cost_{horizon}d"] = (future_ret > estimated_round_trip_cost).astype(float)
        labels.loc[future_ret.isna(), f"y_up_{horizon}d"] = np.nan
        labels.loc[future_ret.isna(), f"y_up_after_cost_{horizon}d"] = np.nan
        labels[f"future_mdd_{horizon}d"] = _future_max_drawdown(close, horizon)
        labels[f"future_vol_{horizon}d"] = _future_volatility(close, horizon)
        mfe, mae = _future_mfe_mae(close, horizon)
        labels[f"future_mfe_{horizon}d"] = mfe
        labels[f"future_mae_{horizon}d"] = mae
        if horizon in benchmark_ret:
            excess = future_ret - benchmark_ret[horizon]
            labels[f"y_excess_ret_{horizon}d"] = excess
            labels[f"y_outperform_index_{horizon}d"] = (excess > 0).astype(float)
            labels.loc[future_ret.isna() | benchmark_ret[horizon].isna(), f"y_outperform_index_{horizon}d"] = np.nan
            labels.loc[future_ret.isna() | benchmark_ret[horizon].isna(), f"y_excess_ret_{horizon}d"] = np.nan

        if horizon == 5:
            good_return = good_trade_return_5d
            good_mae = good_trade_mae_5d
            risk_threshold = crash_threshold_5d
        elif horizon == 20:
            good_return = good_trade_return_20d
            good_mae = good_trade_mae_20d
            risk_threshold = crash_threshold_20d
        else:
            good_return = estimated_round_trip_cost
            good_mae = -abs(estimated_round_trip_cost) * 5.0
            risk_threshold = -abs(estimated_round_trip_cost) * 10.0
        labels[f"y_good_trade_{horizon}d"] = ((future_ret > good_return) & (mae >= good_mae)).astype(float)
        labels[f"y_risk_event_{horizon}d"] = ((labels[f"future_mdd_{horizon}d"] <= risk_threshold) | (labels[f"future_vol_{horizon}d"] > labels[f"future_vol_{horizon}d"].rolling(252, min_periods=60).quantile(0.8))).astype(float)
        missing = future_ret.isna() | mae.isna()
        labels.loc[missing, f"y_good_trade_{horizon}d"] = np.nan
        labels.loc[missing, f"y_risk_event_{horizon}d"] = np.nan

    if "future_mdd_5d" in labels.columns:
        labels["y_crash_5d"] = (labels["future_mdd_5d"] <= crash_threshold_5d).astype(float)
        labels.loc[labels["future_mdd_5d"].isna(), "y_crash_5d"] = np.nan
    if "future_mdd_20d" in labels.columns:
        labels["y_crash_20d"] = (labels["future_mdd_20d"] <= crash_threshold_20d).astype(float)
        labels.loc[labels["future_mdd_20d"].isna(), "y_crash_20d"] = np.nan
    return labels

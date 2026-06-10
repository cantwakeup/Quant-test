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
    good_trade_return_60d: float = 0.12,
    good_trade_mae_5d: float = -0.05,
    good_trade_mae_20d: float = -0.10,
    good_trade_mae_60d: float = -0.18,
    stop_loss_threshold: float = -0.10,
    take_profit_threshold: float = 0.18,
    crash_threshold_5d: float = -0.08,
    crash_threshold_20d: float = -0.15,
) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("prices is empty.")
    prices = prices.reset_index(drop=True)
    label_data: dict[str, pd.Series | np.ndarray] = {"date": prices["date"].values}
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
        y_up = (future_ret > 0).astype(float)
        y_up_after_cost = (future_ret > estimated_round_trip_cost).astype(float)
        y_up[future_ret.isna()] = np.nan
        y_up_after_cost[future_ret.isna()] = np.nan
        future_mdd = _future_max_drawdown(close, horizon)
        future_vol = _future_volatility(close, horizon)
        mfe, mae = _future_mfe_mae(close, horizon)
        label_data[f"y_ret_{horizon}d"] = future_ret
        label_data[f"y_up_{horizon}d"] = y_up
        label_data[f"y_up_after_cost_{horizon}d"] = y_up_after_cost
        label_data[f"future_mdd_{horizon}d"] = future_mdd
        label_data[f"future_vol_{horizon}d"] = future_vol
        label_data[f"future_mfe_{horizon}d"] = mfe
        label_data[f"future_mae_{horizon}d"] = mae
        if horizon in benchmark_ret:
            excess = future_ret - benchmark_ret[horizon]
            outperform = (excess > 0).astype(float)
            excess_missing = future_ret.isna() | benchmark_ret[horizon].isna()
            outperform[excess_missing] = np.nan
            excess[excess_missing] = np.nan
            label_data[f"y_excess_ret_{horizon}d"] = excess
            label_data[f"y_outperform_index_{horizon}d"] = outperform

        if horizon == 5:
            good_return = good_trade_return_5d
            good_mae = good_trade_mae_5d
            risk_threshold = crash_threshold_5d
        elif horizon == 20:
            good_return = good_trade_return_20d
            good_mae = good_trade_mae_20d
            risk_threshold = crash_threshold_20d
        elif horizon == 60:
            good_return = good_trade_return_60d
            good_mae = good_trade_mae_60d
            risk_threshold = crash_threshold_20d * 1.5
        else:
            good_return = estimated_round_trip_cost
            good_mae = -abs(estimated_round_trip_cost) * 5.0
            risk_threshold = -abs(estimated_round_trip_cost) * 10.0
        vol_threshold = future_vol.rolling(252, min_periods=60).quantile(0.8)
        horizon_labels = {
            f"y_good_trade_{horizon}d": ((future_ret > good_return) & (mae >= good_mae)).astype(float),
            f"y_bad_trade_{horizon}d": ((future_ret < -estimated_round_trip_cost) | (mae <= risk_threshold)).astype(float),
            f"y_trend_continue_{horizon}d": ((future_ret > good_return) & (mfe > abs(good_mae))).astype(float),
            f"y_stop_loss_hit_{horizon}d": (mae <= stop_loss_threshold).astype(float),
            f"y_take_profit_before_stop_{horizon}d": ((mfe >= take_profit_threshold) & (mae > stop_loss_threshold)).astype(float),
            f"y_large_drawdown_{horizon}d": (future_mdd <= risk_threshold).astype(float),
            f"y_high_vol_next_{horizon}d": (future_vol > vol_threshold).astype(float),
            f"y_risk_event_{horizon}d": ((future_mdd <= risk_threshold) | (future_vol > vol_threshold)).astype(float),
        }
        if "limit_down" in prices.columns:
            limit_down = prices["limit_down"].astype(float)
            horizon_labels[f"y_limit_down_next_{horizon}d"] = limit_down.shift(-1).rolling(horizon).max().shift(-(horizon - 1))
        if "is_suspended" in prices.columns:
            suspended = prices["is_suspended"].astype(float)
            horizon_labels[f"y_untradable_next_{horizon}d"] = suspended.shift(-1).rolling(horizon).max().shift(-(horizon - 1))
        gap_down = (prices["open"].astype(float).shift(-1) / close - 1.0) <= -0.03
        horizon_labels[f"y_gap_down_next_{horizon}d"] = gap_down.astype(float).shift(-1).rolling(horizon).max().shift(-(horizon - 1))
        missing = future_ret.isna() | mae.isna()
        for col, values in horizon_labels.items():
            values = values.astype(float)
            values[missing] = np.nan
            label_data[col] = values

    if "future_mdd_5d" in label_data:
        crash_5d = (label_data["future_mdd_5d"] <= crash_threshold_5d).astype(float)
        crash_5d[label_data["future_mdd_5d"].isna()] = np.nan
        label_data["y_crash_5d"] = crash_5d
    if "future_mdd_20d" in label_data:
        crash_20d = (label_data["future_mdd_20d"] <= crash_threshold_20d).astype(float)
        crash_20d[label_data["future_mdd_20d"].isna()] = np.nan
        label_data["y_crash_20d"] = crash_20d
    return pd.DataFrame(label_data).copy()

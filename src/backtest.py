from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import performance_metrics


def moving_average_signal(prices: pd.DataFrame, short_window: int = 20, long_window: int = 60) -> pd.DataFrame:
    data = prices[["date", "close"]].copy()
    data["short_ma"] = data["close"].rolling(short_window).mean()
    data["long_ma"] = data["close"].rolling(long_window).mean()
    data["target_position"] = (data["short_ma"] > data["long_ma"]).astype(float)
    return data[["date", "target_position"]]


def simple_momentum_signal(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    data = prices[["date", "close"]].copy()
    data["target_position"] = (data["close"] / data["close"].shift(window) - 1.0 > 0).astype(float)
    return data[["date", "target_position"]]


def volatility_filter_signal(prices: pd.DataFrame, trend_window: int = 60, vol_window: int = 20, vol_quantile: float = 0.7) -> pd.DataFrame:
    data = prices[["date", "close"]].copy()
    ret = data["close"].pct_change()
    trend = data["close"] / data["close"].rolling(trend_window).mean() - 1.0
    vol = ret.rolling(vol_window).std(ddof=0)
    vol_cut = vol.rolling(252, min_periods=60).quantile(vol_quantile)
    data["target_position"] = ((trend > 0) & (vol <= vol_cut)).astype(float)
    return data[["date", "target_position"]]


def relative_strength_signal(prices: pd.DataFrame, benchmark: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    data = prices[["date", "close"]].merge(
        benchmark[["date", "close"]].rename(columns={"close": "benchmark_close"}),
        on="date",
        how="left",
    )
    stock_ret = data["close"] / data["close"].shift(window) - 1.0
    bench_ret = data["benchmark_close"].ffill() / data["benchmark_close"].ffill().shift(window) - 1.0
    data["target_position"] = ((stock_ret > bench_ret) & (stock_ret > 0)).astype(float)
    return data[["date", "target_position"]]


def extract_trade_log(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    open_trade = None
    position = backtest["position"].fillna(0.0)
    for idx, row in backtest.iterrows():
        prev = position.iloc[idx - 1] if idx > 0 else 0.0
        pos = position.iloc[idx]
        if prev <= 0 and pos > 0:
            open_trade = {
                "entry_date": row["date"],
                "entry_price": row["open"],
                "entry_index": idx,
                "entry_position": pos,
            }
        elif prev > 0 and pos <= 0 and open_trade is not None:
            path = backtest.iloc[open_trade["entry_index"] : idx + 1]
            trade_return = path["strategy_return"].add(1.0).prod() - 1.0
            rows.append(
                {
                    **open_trade,
                    "exit_date": row["date"],
                    "exit_price": row["open"],
                    "holding_days": int(idx - open_trade["entry_index"]),
                    "trade_return": float(trade_return),
                    "max_adverse_return": float((path["close"] / open_trade["entry_price"] - 1.0).min()),
                    "max_favorable_return": float((path["close"] / open_trade["entry_price"] - 1.0).max()),
                }
            )
            open_trade = None
    return pd.DataFrame(rows)


def annual_return_table(backtest: pd.DataFrame, return_col: str = "strategy_return") -> pd.DataFrame:
    data = backtest[["date", return_col]].copy()
    data["year"] = pd.to_datetime(data["date"]).dt.year
    return data.groupby("year")[return_col].apply(lambda x: (1.0 + x.fillna(0.0)).prod() - 1.0).reset_index(name="return")


def regime_return_table(backtest: pd.DataFrame, prices: pd.DataFrame, return_col: str = "strategy_return") -> pd.DataFrame:
    data = backtest[["date", return_col]].merge(prices[["date", "close"]], on="date", how="left")
    ret60 = data["close"] / data["close"].shift(60) - 1.0
    vol20 = data["close"].pct_change().rolling(20).std(ddof=0)
    vol_cut = vol20.rolling(252, min_periods=60).median()
    data["regime"] = np.where(ret60 > 0.15, "uptrend", np.where(ret60 < -0.15, "downtrend", "sideways"))
    data["vol_state"] = np.where(vol20 > vol_cut, "high_vol", "low_vol")
    rows = []
    for key in ("regime", "vol_state"):
        for name, group in data.groupby(key):
            rows.append({"segment": key, "state": name, "return": float((1.0 + group[return_col].fillna(0.0)).prod() - 1.0), "days": int(len(group))})
    return pd.DataFrame(rows)


def run_vector_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    benchmark: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or {}
    commission = float(cfg.get("commission", 0.0003))
    stamp_tax = float(cfg.get("stamp_tax", 0.001))
    slippage = float(cfg.get("slippage", 0.0005))
    minimum_fee = float(cfg.get("minimum_fee", 0.0))
    max_position = float(cfg.get("max_position", 1.0))
    stop_loss = float(cfg.get("stop_loss", 0.10))
    trailing_stop = float(cfg.get("trailing_stop", 0.15))
    max_holding_days = int(cfg.get("max_holding_days", 60))
    limit_aware = bool(cfg.get("limit_aware", True))

    data = prices[["date", "open", "close", "limit_up", "limit_down", "is_suspended"]].copy()
    data = data.merge(signals[["date", "target_position"]], on="date", how="left")
    data["target_position"] = data["target_position"].fillna(0.0).clip(0.0, max_position)
    data = data.sort_values("date").reset_index(drop=True)

    n = len(data)
    position = np.zeros(n, dtype=float)
    costs = np.zeros(n, dtype=float)
    turnover = np.zeros(n, dtype=float)
    forced_exit_next = False
    entry_price = np.nan
    peak_price = np.nan
    holding_days = 0

    for i in range(n):
        prev_pos = position[i - 1] if i > 0 else 0.0
        desired = data.loc[i - 1, "target_position"] if i > 0 else 0.0
        if forced_exit_next:
            desired = 0.0
            forced_exit_next = False

        can_trade = True
        if limit_aware:
            if bool(data.loc[i, "is_suspended"]):
                can_trade = False
            if desired > prev_pos and bool(data.loc[i, "limit_up"]):
                can_trade = False
            if desired < prev_pos and bool(data.loc[i, "limit_down"]):
                can_trade = False

        pos = desired if can_trade else prev_pos
        position[i] = pos
        buy_turnover = max(pos - prev_pos, 0.0)
        sell_turnover = max(prev_pos - pos, 0.0)
        turnover[i] = buy_turnover + sell_turnover
        variable_cost = (buy_turnover + sell_turnover) * (commission + slippage) + sell_turnover * stamp_tax
        costs[i] = variable_cost + (minimum_fee if (buy_turnover + sell_turnover) > 0 else 0.0)

        close = float(data.loc[i, "close"])
        if pos > 0 and prev_pos <= 0:
            entry_price = close
            peak_price = close
            holding_days = 0
        elif pos > 0:
            holding_days += 1
            peak_price = max(peak_price, close) if np.isfinite(peak_price) else close
        else:
            entry_price = np.nan
            peak_price = np.nan
            holding_days = 0

        if pos > 0 and np.isfinite(entry_price):
            stop_hit = close / entry_price - 1.0 <= -stop_loss
            trail_hit = np.isfinite(peak_price) and close / peak_price - 1.0 <= -trailing_stop
            time_hit = holding_days >= max_holding_days
            if stop_hit or trail_hit or time_hit:
                forced_exit_next = True

    data["position"] = position
    data["turnover"] = turnover
    data["transaction_cost"] = costs
    data["asset_return"] = data["open"].shift(-1) / data["open"] - 1.0
    data["asset_return"] = data["asset_return"].fillna(0.0)
    data["strategy_gross_return"] = data["position"] * data["asset_return"]
    data["strategy_return"] = data["strategy_gross_return"] - data["transaction_cost"]
    data["buy_hold_return"] = data["asset_return"]
    data["cash_return"] = 0.0
    data["strategy_equity"] = (1.0 + data["strategy_return"]).cumprod()
    data["strategy_gross_equity"] = (1.0 + data["strategy_gross_return"]).cumprod()
    data["buy_hold_equity"] = (1.0 + data["buy_hold_return"]).cumprod()
    data["cash_equity"] = 1.0

    if benchmark is not None and not benchmark.empty:
        bench = benchmark[["date", "close"]].rename(columns={"close": "benchmark_close"})
        data = data.merge(bench, on="date", how="left")
        data["benchmark_close"] = data["benchmark_close"].ffill()
        data["benchmark_return"] = data["benchmark_close"].shift(-1) / data["benchmark_close"] - 1.0
        data["benchmark_return"] = data["benchmark_return"].fillna(0.0)
        data["benchmark_equity"] = (1.0 + data["benchmark_return"]).cumprod()
    else:
        data["benchmark_return"] = np.nan
        data["benchmark_equity"] = np.nan

    strategy_metrics = performance_metrics(
        data["strategy_return"],
        data["strategy_equity"],
        benchmark_returns=data["buy_hold_return"],
        position=data["position"],
    )
    buy_hold_metrics = performance_metrics(data["buy_hold_return"], data["buy_hold_equity"])
    gross_metrics = performance_metrics(data["strategy_gross_return"], data["strategy_gross_equity"], position=data["position"])
    cash_metrics = performance_metrics(data["cash_return"], data["cash_equity"])
    rows = []
    for name, metrics in (
        ("strategy", strategy_metrics),
        ("strategy_before_cost", gross_metrics),
        ("buy_hold_300316", buy_hold_metrics),
        ("cash", cash_metrics),
    ):
        rows.append({"portfolio": name, **metrics})
    if data["benchmark_equity"].notna().any():
        rows.append({"portfolio": "benchmark_index", **performance_metrics(data["benchmark_return"], data["benchmark_equity"])})
    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        bh_total = metrics_df.loc[metrics_df["portfolio"] == "buy_hold_300316", "total_return"]
        if len(bh_total):
            metrics_df["excess_vs_buy_hold"] = metrics_df["total_return"] - float(bh_total.iloc[0])
    return data, metrics_df


def run_baseline_backtests(prices: pd.DataFrame, benchmark: Optional[pd.DataFrame], config: Dict[str, object]) -> pd.DataFrame:
    baselines = {
        "empty_no_trade": pd.DataFrame({"date": prices["date"], "target_position": 0.0}),
        "ma_20_60": moving_average_signal(prices, 20, 60),
        "ma_60_120": moving_average_signal(prices, 60, 120),
        "momentum_20": simple_momentum_signal(prices, 20),
        "volatility_filter": volatility_filter_signal(prices),
    }
    if benchmark is not None and not benchmark.empty:
        baselines["relative_strength_20"] = relative_strength_signal(prices, benchmark, 20)
    rows = []
    for name, signal in baselines.items():
        _, metrics = run_vector_backtest(prices, signal, benchmark=benchmark, config=config)
        metric = metrics.loc[metrics["portfolio"] == "strategy"].copy()
        if not metric.empty:
            metric.insert(0, "baseline", name)
            rows.append(metric)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .utils import rolling_max_drawdown


@dataclass(frozen=True)
class StrategySpec:
    name: str
    kind: str
    fast: Optional[int] = None
    slow: Optional[int] = None
    window: Optional[int] = None
    risk_filter: str = "none"


def risk_filter_mask(prices: pd.DataFrame, filter_name: str) -> pd.Series:
    close = prices["close"].astype(float)
    amount = prices.get("amount", pd.Series(np.nan, index=prices.index)).astype(float)
    ret = close.pct_change()
    if filter_name == "none":
        return pd.Series(True, index=prices.index)
    if filter_name == "close_gt_ma120":
        return close > close.rolling(120).mean()
    if filter_name == "close_gt_ma250":
        return close > close.rolling(250).mean()
    if filter_name == "vol_pct_lt_80":
        vol20 = ret.rolling(20).std(ddof=0)
        cut = vol20.rolling(252, min_periods=60).quantile(0.8)
        return vol20 < cut
    if filter_name == "mdd60_gt_-25":
        return rolling_max_drawdown(close, 60) > -0.25
    if filter_name == "amount_liquidity":
        cut = amount.rolling(252, min_periods=60).median()
        return amount >= cut
    if filter_name == "gap_risk":
        gap_abs = (prices["open"].astype(float) / close.shift(1) - 1.0).abs()
        return gap_abs.rolling(20).mean() < 0.025
    if filter_name == "limit_tradable":
        return ~(prices.get("limit_up", False).astype(bool) | prices.get("limit_down", False).astype(bool) | prices.get("is_suspended", False).astype(bool))
    raise ValueError(f"Unknown risk filter: {filter_name}")


def ma_cross_signal(prices: pd.DataFrame, fast: int, slow: int, risk_filter: str = "none") -> pd.DataFrame:
    if fast >= slow:
        raise ValueError("fast MA must be below slow MA")
    close = prices["close"].astype(float)
    signal = (close.rolling(fast).mean() > close.rolling(slow).mean()) & risk_filter_mask(prices, risk_filter)
    return pd.DataFrame({"date": prices["date"], "target_position": signal.astype(float)})


def momentum_signal(prices: pd.DataFrame, window: int, risk_filter: str = "none") -> pd.DataFrame:
    close = prices["close"].astype(float)
    signal = (close / close.shift(window) - 1.0 > 0) & risk_filter_mask(prices, risk_filter)
    return pd.DataFrame({"date": prices["date"], "target_position": signal.astype(float)})


def breakout_signal(prices: pd.DataFrame, window: int = 120, risk_filter: str = "none") -> pd.DataFrame:
    close = prices["close"].astype(float)
    high = prices["high"].astype(float)
    signal = (close > high.shift(1).rolling(window).max()) & risk_filter_mask(prices, risk_filter)
    return pd.DataFrame({"date": prices["date"], "target_position": signal.astype(float)})


def build_signal_from_spec(prices: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    if spec.kind == "ma":
        return ma_cross_signal(prices, int(spec.fast), int(spec.slow), spec.risk_filter)
    if spec.kind == "momentum":
        return momentum_signal(prices, int(spec.window), spec.risk_filter)
    if spec.kind == "breakout":
        return breakout_signal(prices, int(spec.window or 120), spec.risk_filter)
    if spec.kind == "cash":
        return pd.DataFrame({"date": prices["date"], "target_position": 0.0})
    if spec.kind == "buy_hold":
        return pd.DataFrame({"date": prices["date"], "target_position": 1.0})
    raise ValueError(f"Unknown strategy kind: {spec.kind}")


def default_strategy_specs() -> list[StrategySpec]:
    return [
        StrategySpec("cash", "cash"),
        StrategySpec("buy_hold", "buy_hold"),
        StrategySpec("ma_60_120", "ma", fast=60, slow=120),
        StrategySpec("ma_20_60", "ma", fast=20, slow=60),
        StrategySpec("momentum_20", "momentum", window=20),
        StrategySpec("momentum_60", "momentum", window=60),
        StrategySpec("breakout_120", "breakout", window=120),
        StrategySpec("trend_follow_with_vol_filter", "ma", fast=60, slow=120, risk_filter="vol_pct_lt_80"),
        StrategySpec("trend_follow_with_drawdown_filter", "ma", fast=60, slow=120, risk_filter="mdd60_gt_-25"),
    ]

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .data_cleaning import align_fundamentals_by_announcement
from .utils import rolling_max_drawdown, rolling_slope, safe_divide


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(window).mean()
    losses = (-delta.clip(upper=0)).rolling(window).mean()
    rs = safe_divide(gains, losses)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close: pd.Series) -> pd.DataFrame:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    return pd.DataFrame({"macd_dif": dif, "macd_dea": dea, "macd_hist": hist})


def _kdj(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 9) -> pd.DataFrame:
    low_n = low.rolling(window).min()
    high_n = high.rolling(window).max()
    rsv = safe_divide(close - low_n, high_n - low_n) * 100.0
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    atr = _atr(high, low, close, window)
    plus_di = 100.0 * safe_divide(plus_dm.rolling(window).mean(), atr)
    minus_di = 100.0 * safe_divide(minus_dm.rolling(window).mean(), atr)
    dx = 100.0 * safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    return dx.rolling(window).mean()


def _downside_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.where(returns < 0, 0.0).rolling(window).std(ddof=0)


def build_price_volume_features(
    prices: pd.DataFrame,
    return_windows: Iterable[int] = (1, 3, 5, 10, 20, 60),
    ma_windows: Iterable[int] = (5, 10, 20, 60, 120),
    breakout_windows: Iterable[int] = (20, 60, 120),
    volatility_windows: Iterable[int] = (5, 10, 20, 60),
    volume_windows: Iterable[int] = (5, 20, 60),
    zscore_windows: Iterable[int] = (20, 60),
) -> pd.DataFrame:
    df = prices.copy()
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    amount = df["amount"].astype(float)
    turnover = df["turnover"].astype(float)
    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))

    features = pd.DataFrame({"date": df["date"]})
    for window in return_windows:
        features[f"ret_{window}d"] = close / close.shift(window) - 1.0
        features[f"log_ret_{window}d"] = np.log(close / close.shift(window))

    for window in ma_windows:
        ma = close.rolling(window).mean()
        features[f"ma_gap_{window}d"] = close / ma - 1.0
        features[f"ma_slope_{window}d"] = rolling_slope(ma, min(window, 20)) if window >= 5 else np.nan
        features[f"price_above_ma_{window}d"] = (close > ma).astype(float)

    for window in breakout_windows:
        past_high = high.shift(1).rolling(window).max()
        past_low = low.shift(1).rolling(window).min()
        features[f"break_high_{window}d"] = (close > past_high).astype(float)
        features[f"break_low_{window}d"] = (close < past_low).astype(float)
        features[f"dist_to_high_{window}d"] = close / past_high - 1.0
        features[f"dist_to_low_{window}d"] = close / past_low - 1.0

    macd = _macd(close)
    kdj = _kdj(high, low, close)
    features = pd.concat([features, macd, kdj], axis=1)
    features["adx_14d"] = _adx(high, low, close, 14)
    features["rsi_6d"] = _rsi(close, 6)
    features["rsi_14d"] = _rsi(close, 14)

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    upper = ma20 + 2.0 * std20
    lower = ma20 - 2.0 * std20
    features["boll_pos_20d"] = safe_divide(close - lower, upper - lower)
    for window in zscore_windows:
        mean = close.rolling(window).mean()
        std = close.rolling(window).std(ddof=0)
        features[f"close_zscore_{window}d"] = safe_divide(close - mean, std)

    for window in volume_windows:
        vol_ma = volume.rolling(window).mean()
        amt_ma = amount.rolling(window).mean()
        features[f"volume_ratio_{window}d"] = safe_divide(volume, vol_ma)
        features[f"amount_ratio_{window}d"] = safe_divide(amount, amt_ma)
    features["amount_chg_1d"] = amount.pct_change()
    features["amount_chg_5d"] = amount / amount.shift(5) - 1.0
    features["turnover_chg_1d"] = turnover.diff()
    features["turnover_ratio_20d"] = safe_divide(turnover, turnover.rolling(20).mean())
    features["volume_up_price_up"] = ((ret > 0) & (volume > volume.rolling(20).mean())).astype(float)
    features["volume_up_price_down"] = ((ret < 0) & (volume > volume.rolling(20).mean())).astype(float)
    features["shrink_pullback"] = ((ret < 0) & (volume < volume.rolling(20).mean())).astype(float)
    features["price_volume_divergence_20d"] = ret.rolling(20).corr(volume.pct_change())

    for window in volatility_windows:
        features[f"volatility_{window}d"] = log_ret.rolling(window).std(ddof=0)
        features[f"downside_vol_{window}d"] = _downside_volatility(log_ret, window)
        features[f"max_drawdown_{window}d"] = rolling_max_drawdown(close, window)
    features["atr_14d"] = _atr(high, low, close, 14)
    features["atr_pct_14d"] = safe_divide(features["atr_14d"], close)
    features["intraday_range"] = safe_divide(high - low, close)
    features["high_low_range_20d"] = safe_divide(high.rolling(20).max() - low.rolling(20).min(), close)
    features["intraday_strength"] = safe_divide(2.0 * close - high - low, high - low)
    features["kbar_body"] = safe_divide(close - open_, open_)
    return features.replace([np.inf, -np.inf], np.nan)


def add_relative_strength_features(features: pd.DataFrame, prices: pd.DataFrame, market: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if market is None or market.empty:
        return features
    merged = prices[["date", "close", "amount"]].merge(
        market[["date", "close", "amount"]].rename(columns={"close": f"{prefix}_close", "amount": f"{prefix}_amount"}),
        on="date",
        how="left",
    )
    stock_close = merged["close"]
    market_close = merged[f"{prefix}_close"].ffill()
    stock_amount = merged["amount"]
    market_amount = merged[f"{prefix}_amount"].ffill()
    for window in (1, 3, 5, 10, 20, 60):
        stock_ret = stock_close / stock_close.shift(window) - 1.0
        market_ret = market_close / market_close.shift(window) - 1.0
        features[f"rs_{prefix}_{window}d"] = stock_ret - market_ret
    features[f"{prefix}_trend_20d"] = market_close / market_close.rolling(20).mean() - 1.0
    features[f"{prefix}_volatility_20d"] = np.log(market_close / market_close.shift(1)).rolling(20).std(ddof=0)
    features[f"amount_rs_{prefix}_20d"] = (stock_amount / stock_amount.shift(20) - 1.0) - (
        market_amount / market_amount.shift(20) - 1.0
    )
    return features


def build_fundamental_features(aligned: pd.DataFrame) -> pd.DataFrame:
    if aligned is None or aligned.empty:
        return pd.DataFrame()
    result = pd.DataFrame({"date": aligned["date"]})
    skip = {"date", "ann_date", "report_date", "ts_code", "symbol", "report_type"}
    numeric_cols = [c for c in aligned.columns if c not in skip and pd.api.types.is_numeric_dtype(aligned[c])]
    change_mask = aligned["ann_date"].ne(aligned["ann_date"].shift()) if "ann_date" in aligned.columns else pd.Series(False, index=aligned.index)
    for col in numeric_cols:
        series = aligned[col].astype(float)
        result[f"fund_{col}"] = series
        event_values = series.where(change_mask)
        result[f"fund_{col}_chg"] = event_values.pct_change().ffill()
    if {"net_cash_flows_oper_act", "n_income_attr_p"}.issubset(aligned.columns):
        result["fund_ocf_to_profit"] = safe_divide(aligned["net_cash_flows_oper_act"], aligned["n_income_attr_p"])
    if "ann_date" in aligned.columns:
        result["days_since_fin_report"] = (aligned["date"] - aligned["ann_date"]).dt.days
        for window in (5, 20):
            result[f"post_fin_report_{window}d"] = (result["days_since_fin_report"].between(0, window)).astype(float)
    return result.replace([np.inf, -np.inf], np.nan)


def build_event_features(dates: pd.Series, events: Optional[pd.DataFrame]) -> pd.DataFrame:
    features = pd.DataFrame({"date": dates})
    if events is None or events.empty:
        return features
    event_df = events.copy()
    event_df["event_date"] = pd.to_datetime(event_df["event_date"])
    event_df["event_type"] = event_df["event_type"].fillna("generic_event").astype(str)
    for event_type, group in event_df.groupby("event_type"):
        clean_type = "".join(ch if ch.isalnum() else "_" for ch in event_type.lower())
        marker = group[["event_date"]].drop_duplicates().sort_values("event_date")
        marker[f"last_{clean_type}_date"] = marker["event_date"]
        aligned = pd.merge_asof(
            pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date"),
            marker,
            left_on="date",
            right_on="event_date",
            direction="backward",
        )
        days = (aligned["date"] - aligned[f"last_{clean_type}_date"]).dt.days
        features[f"days_since_{clean_type}"] = days
        for window in (5, 20):
            features[f"post_{clean_type}_{window}d"] = days.between(0, window).astype(float)
    return features


def build_feature_table(
    prices: pd.DataFrame,
    feature_config: Optional[Dict[str, object]] = None,
    market_indices: Optional[Dict[str, pd.DataFrame]] = None,
    industry: Optional[pd.DataFrame] = None,
    peer_index: Optional[pd.DataFrame] = None,
    fundamentals: Optional[pd.DataFrame] = None,
    events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    cfg = feature_config or {}
    features = build_price_volume_features(
        prices,
        return_windows=cfg.get("return_windows", (1, 3, 5, 10, 20, 60)),
        ma_windows=cfg.get("ma_windows", (5, 10, 20, 60, 120)),
        breakout_windows=cfg.get("breakout_windows", (20, 60, 120)),
        volatility_windows=cfg.get("volatility_windows", (5, 10, 20, 60)),
        volume_windows=cfg.get("volume_windows", (5, 20, 60)),
        zscore_windows=cfg.get("zscore_windows", (20, 60)),
    )
    for name, market in (market_indices or {}).items():
        features = add_relative_strength_features(features, prices, market, prefix=name)
    if industry is not None and not industry.empty:
        features = add_relative_strength_features(features, prices, industry, prefix="industry")
    if peer_index is not None and not peer_index.empty:
        features = add_relative_strength_features(features, prices, peer_index, prefix="peers")

    aligned_fin = align_fundamentals_by_announcement(prices, fundamentals) if fundamentals is not None else pd.DataFrame()
    fund_features = build_fundamental_features(aligned_fin)
    event_features = build_event_features(prices["date"], events)
    for extra in (fund_features, event_features):
        if extra is not None and not extra.empty:
            features = features.merge(extra, on="date", how="left")
    return features.replace([np.inf, -np.inf], np.nan)

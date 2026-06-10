from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def clean_ohlcv(
    data: pd.DataFrame,
    start: Optional[object] = None,
    end: Optional[object] = None,
) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    df = data.copy()
    if "trade_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"trade_date": "date"})
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    if "date" not in df.columns:
        raise ValueError("OHLCV data must contain a date column.")
    df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
    df = df.dropna(subset=["date"])

    for column in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_change", "change", "amplitude"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ["open", "high", "low", "close"]:
        if column not in df.columns:
            raise ValueError(f"OHLCV data missing required column: {column}")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    if "amount" not in df.columns:
        df["amount"] = np.nan
    if "turnover" not in df.columns:
        df["turnover"] = np.nan
    if "pct_change" not in df.columns:
        df["pct_change"] = df["close"].pct_change() * 100.0

    df = df.sort_values("date").drop_duplicates("date", keep="last")
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    df["is_suspended"] = df["close"].isna() | (df["volume"].fillna(0) <= 0)
    df["limit_up"] = df["pct_change"] >= 19.8
    df["limit_down"] = df["pct_change"] <= -19.8
    df["adj_close"] = df["adj_close"] if "adj_close" in df.columns else df["close"]
    return df.reset_index(drop=True)


def align_fundamentals_by_announcement(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """As-of merge financial rows by actual announcement date.

    This avoids using report-period end dates before the disclosure was known.
    """
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame({"date": prices["date"]})
    fin = fundamentals.copy()
    if "ann_date" not in fin.columns:
        if "announcement_date" in fin.columns:
            fin = fin.rename(columns={"announcement_date": "ann_date"})
        else:
            raise ValueError("Fundamental data must contain ann_date or announcement_date.")
    fin["ann_date"] = pd.to_datetime(fin["ann_date"])
    if "report_date" in fin.columns:
        fin["report_date"] = pd.to_datetime(fin["report_date"])
    elif "end_date" in fin.columns:
        fin = fin.rename(columns={"end_date": "report_date"})
        fin["report_date"] = pd.to_datetime(fin["report_date"])
    else:
        fin["report_date"] = pd.NaT

    date_frame = prices[["date"]].copy().sort_values("date")
    fin = fin.sort_values("ann_date")
    aligned = pd.merge_asof(date_frame, fin, left_on="date", right_on="ann_date", direction="backward")
    if "ann_date" in aligned.columns:
        leaked = aligned["ann_date"].notna() & (aligned["ann_date"] > aligned["date"])
        if leaked.any():
            raise AssertionError("Fundamental announcement alignment leaked future data.")
    return aligned


def validate_feature_dates(features: pd.DataFrame, prices: pd.DataFrame) -> None:
    if "date" not in features.columns:
        raise ValueError("Feature table must contain date.")
    left = pd.to_datetime(features["date"])
    right = pd.to_datetime(prices["date"])
    if not left.is_monotonic_increasing:
        raise AssertionError("Feature dates must be sorted.")
    if not left.isin(set(right)).all():
        raise AssertionError("Feature table contains dates not in price data.")


def make_peer_index(peer_data: dict) -> pd.DataFrame:
    frames = []
    for symbol, df in (peer_data or {}).items():
        if df is None or df.empty:
            continue
        tmp = df[["date", "close", "amount"]].copy()
        tmp = tmp.rename(columns={"close": f"{symbol}_close", "amount": f"{symbol}_amount"})
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    close_cols = [c for c in merged.columns if c.endswith("_close")]
    amount_cols = [c for c in merged.columns if c.endswith("_amount")]
    for col in close_cols:
        merged[col] = merged[col] / merged[col].dropna().iloc[0]
    merged["close"] = merged[close_cols].mean(axis=1)
    merged["amount"] = merged[amount_cols].mean(axis=1) if amount_cols else np.nan
    return merged[["date", "close", "amount"]].sort_values("date").reset_index(drop=True)

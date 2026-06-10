from __future__ import annotations

from typing import Dict

import pandas as pd

from .industry_mapping import PEER_GROUPS


def build_peer_baskets(peer_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    baskets: Dict[str, pd.DataFrame] = {}
    for group, members in PEER_GROUPS.items():
        frames = []
        for symbol in members:
            data = peer_data.get(symbol)
            if data is None or data.empty or "close" not in data.columns:
                continue
            tmp = data[["date", "close", "amount"]].copy()
            tmp["norm_close"] = tmp["close"] / tmp["close"].dropna().iloc[0]
            tmp = tmp.rename(columns={"norm_close": symbol, "amount": f"{symbol}_amount"})
            frames.append(tmp[["date", symbol, f"{symbol}_amount"]])
        if not frames:
            continue
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="date", how="outer")
        close_cols = [c for c in merged.columns if c in members]
        amount_cols = [c for c in merged.columns if c.endswith("_amount")]
        baskets[group] = pd.DataFrame(
            {
                "date": merged["date"],
                "close": merged[close_cols].mean(axis=1),
                "amount": merged[amount_cols].mean(axis=1) if amount_cols else pd.NA,
            }
        ).sort_values("date")
    return baskets

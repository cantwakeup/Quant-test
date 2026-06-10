from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .data_adapter import EastmoneyAdapter
from .data_cleaning import clean_ohlcv


INDEX_SYMBOLS = {
    "hs300": "000300.SH",
    "zz500": "000905.SH",
    "zz1000": "000852.SH",
    "chinext": "399006.SZ",
    "sz_comp": "399001.SZ",
}


def try_update_stock_daily(symbol: str, output_path: Path, start: str, end: Optional[str] = None, adjust: str = "qfq") -> Dict[str, object]:
    """Try Eastmoney download; keep existing local CSV when network fails."""
    result: Dict[str, object] = {
        "dataset": f"stock_daily_{symbol}",
        "output_path": str(output_path),
        "attempted_source": "eastmoney",
        "success": False,
        "error": "",
    }
    try:
        data = EastmoneyAdapter().get_stock_daily(symbol, start=start, end=end or date.today(), adjust=adjust)
        if data.empty:
            raise ValueError("download returned empty data")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False, encoding="utf-8-sig")
        result.update({"success": True, "rows": len(data), "start_date": str(data["date"].min().date()), "end_date": str(data["date"].max().date())})
    except Exception as exc:  # network may be unavailable; this is expected in offline runs.
        result["error"] = repr(exc)
    return result


def read_local_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    if data.empty:
        return data
    if {"open", "high", "low", "close"}.issubset(data.columns):
        return clean_ohlcv(data)
    return data

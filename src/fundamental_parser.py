from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


ANN_DATE_COLUMNS = ("ann_date", "announcement_date")


def normalize_financial_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    df = data.copy()
    ann_col = next((c for c in ANN_DATE_COLUMNS if c in df.columns), None)
    if ann_col is None:
        df["usable_for_model"] = False
        df["unusable_reason"] = "missing announcement date"
        return df
    if ann_col != "ann_date":
        df = df.rename(columns={ann_col: "ann_date"})
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["usable_for_model"] = df["ann_date"].notna()
    df["unusable_reason"] = np.where(df["usable_for_model"], "", "invalid announcement date")
    return df.sort_values("ann_date")


def available_financial_fields(data: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    rows = []
    for field in required:
        rows.append({"field": field, "available": bool(data is not None and field in data.columns and data[field].notna().any())})
    return pd.DataFrame(rows)

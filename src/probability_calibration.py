from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(prob: pd.Series, target: pd.Series) -> float:
    sample = pd.concat([prob, target], axis=1).dropna()
    if sample.empty:
        return np.nan
    return float(((sample.iloc[:, 0].clip(0, 1) - sample.iloc[:, 1].clip(0, 1)) ** 2).mean())


def calibration_table(prob: pd.Series, target: pd.Series, bins: int = 5) -> pd.DataFrame:
    sample = pd.concat([prob, target], axis=1).dropna()
    sample.columns = ["prob", "target"]
    if sample.empty or sample["prob"].nunique() < 2:
        return pd.DataFrame()
    sample["bin"] = pd.qcut(sample["prob"].rank(method="first"), bins, labels=False, duplicates="drop") + 1
    return sample.groupby("bin").agg(mean_prob=("prob", "mean"), event_rate=("target", "mean"), count=("target", "size")).reset_index()

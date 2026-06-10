from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .probability_calibration import brier_score, calibration_table


def evaluate_meta_tasks(predictions: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = predictions.merge(labels, on="date", how="inner")
    rows = []
    calibration_rows = []
    tasks = {
        "good_trade_20d": ("prob_up_20d", "y_good_trade_20d"),
        "bad_trade_proxy_20d": ("prob_up_20d", "y_risk_event_20d"),
        "up_after_cost_20d": ("prob_up_20d", "y_up_after_cost_20d"),
    }
    for task, (prob_col, target_col) in tasks.items():
        if prob_col not in merged.columns or target_col not in merged.columns:
            continue
        prob = merged[prob_col]
        target = merged[target_col]
        sample = pd.concat([prob, target], axis=1).dropna()
        if sample.empty:
            continue
        rows.append(
            {
                "task": task,
                "model": "numpy_logistic_proxy",
                "observations": len(sample),
                "brier": brier_score(prob, target),
                "accuracy": ((prob >= 0.5) == (target > 0.5)).mean(),
                "positive_rate": target.mean(),
                "adopt_as_primary": False,
                "reason": "proxy probability does not yet beat simple trend rules consistently",
            }
        )
        cal = calibration_table(prob, target)
        if not cal.empty:
            cal.insert(0, "task", task)
            calibration_rows.append(cal)
    return pd.DataFrame(rows), pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()

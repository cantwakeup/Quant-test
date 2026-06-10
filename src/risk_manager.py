from __future__ import annotations

import numpy as np
import pandas as pd


def risk_state(row: pd.Series) -> tuple[str, str]:
    risk_score = row.get("risk_score", np.nan)
    if pd.notna(risk_score) and risk_score >= 0.9:
        return "risk_off", "risk_score above 0.9"
    if bool(row.get("manual_review_required", False)):
        return "manual_review", "data quality requires manual review"
    return "normal", "risk within configured bounds"


def cap_position_for_risk(target: float, risk_score: float) -> float:
    if pd.isna(risk_score):
        return 0.0
    if risk_score >= 0.9:
        return 0.0
    if risk_score >= 0.7:
        return min(target, 0.25)
    if risk_score >= 0.55:
        return min(target, 0.5)
    return target

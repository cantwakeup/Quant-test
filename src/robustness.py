from __future__ import annotations

import pandas as pd


def robustness_decision(row: pd.Series) -> tuple[bool, str, bool]:
    if row.get("total_return", -999) <= 0:
        return False, "non-positive return under perturbation", False
    if row.get("max_drawdown", -1) < -0.45:
        return False, "drawdown too large", False
    if row.get("beats_buy_hold", False) is False:
        return False, "does not beat buy and hold", False
    return True, "passes basic robustness screen", True


def annotate_robustness(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    annotations = result.apply(lambda row: robustness_decision(row), axis=1)
    result["robust_pass"] = [a[0] for a in annotations]
    result["fail_reason"] = [a[1] for a in annotations]
    result["whether_candidate_for_paper_trade"] = [a[2] for a in annotations]
    return result

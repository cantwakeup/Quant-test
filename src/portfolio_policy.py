from __future__ import annotations

import pandas as pd

from .risk_manager import cap_position_for_risk


def policy_action(current_position: float, target_position: float) -> str:
    if target_position > current_position:
        return "increase"
    if target_position < current_position:
        return "reduce"
    if target_position > 0:
        return "hold"
    return "no_trade"


def enrich_policy_fields(signals: pd.DataFrame, current_position: float = 0.0, data_quality_flag: str = "missing_external_data") -> pd.DataFrame:
    result = signals.copy()
    result["current_position"] = current_position
    result["max_position_allowed"] = result["risk_score"].map(lambda x: cap_position_for_risk(1.0, x))
    result["target_position"] = result[["target_position", "max_position_allowed"]].min(axis=1)
    result["action"] = result["target_position"].map(lambda x: policy_action(current_position, x))
    result["action_reason"] = result["reason"]
    result["data_quality_flag"] = data_quality_flag
    result["manual_review_required"] = data_quality_flag != "complete"
    result["next_trade_date"] = result["date"].shift(-1)
    result["trailing_stop_reference"] = result["stop_loss_reference"]
    result["meta_model_prob_good_trade_20d"] = result.get("prob_up_20d")
    result["meta_model_prob_bad_trade_20d"] = 1.0 - result.get("prob_up_20d")
    result["model_score"] = result.get("signal_score")
    result["pred_excess_ret_20d"] = pd.NA
    result["expected_mfe_20d"] = pd.NA
    result["expected_mae_20d"] = pd.NA
    return result

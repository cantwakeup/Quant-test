from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def _bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if pd.isna(value):
        return np.nan
    return float(min(max(value, lower), upper))


def _risk_score(row: pd.Series, cfg: Dict[str, object]) -> float:
    vol_limit = float(cfg.get("risk_off_vol_20d", 0.045))
    dd_limit = abs(float(cfg.get("risk_off_drawdown_20d", -0.15)))
    vol = row.get("volatility_20d", np.nan)
    dd = row.get("max_drawdown_20d", np.nan)
    vol_part = _bounded(vol / vol_limit) if pd.notna(vol) and vol_limit > 0 else 0.5
    dd_part = _bounded(abs(dd) / dd_limit) if pd.notna(dd) and dd_limit > 0 else 0.5
    return float(0.55 * vol_part + 0.45 * dd_part)


def generate_signals(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, object],
    primary_horizons: Iterable[int] = (5, 20),
) -> pd.DataFrame:
    cfg = config or {}
    horizons = list(primary_horizons)
    h5 = 5 if 5 in horizons else horizons[0]
    h20 = 20 if 20 in horizons else horizons[-1]
    data = prices[["date", "close", "limit_up", "limit_down", "is_suspended"]].merge(predictions, on="date", how="left")
    keep_features = [
        "volatility_20d",
        "max_drawdown_20d",
        "ma_gap_20d",
        "rsi_14d",
        "volume_ratio_20d",
    ]
    data = data.merge(features[["date"] + [c for c in keep_features if c in features.columns]], on="date", how="left")

    max_position = float(cfg.get("max_position", 1.0))
    entry = float(cfg.get("entry_threshold", 0.56))
    strong_entry = float(cfg.get("strong_entry_threshold", 0.62))
    exit_threshold = float(cfg.get("exit_threshold", 0.48))
    min_ret_5 = float(cfg.get("min_expected_return_5d", 0.01))
    min_ret_20 = float(cfg.get("min_expected_return_20d", 0.025))

    rows = []
    for _, row in data.iterrows():
        pred5 = row.get(f"pred_ret_{h5}d", np.nan)
        pred20 = row.get(f"pred_ret_{h20}d", np.nan)
        prob5 = row.get(f"prob_up_{h5}d", np.nan)
        prob20 = row.get(f"prob_up_{h20}d", np.nan)
        risk = _risk_score(row, cfg)

        if pd.isna(prob5) or pd.isna(prob20) or pd.isna(pred5) or pd.isna(pred20):
            label = "no_trade"
            target = 0.0
            score = np.nan
            reason = "walk-forward prediction unavailable"
            warning = "data insufficient for model signal"
        elif bool(row.get("is_suspended", False)):
            label = "no_trade"
            target = 0.0
            score = 0.0
            reason = "suspension flag"
            warning = "cannot assume executable trade"
        else:
            score = (
                0.35 * ((prob5 - 0.5) / 0.15)
                + 0.25 * ((prob20 - 0.5) / 0.15)
                + 0.20 * (pred5 / max(abs(min_ret_5), 1e-6))
                + 0.20 * (pred20 / max(abs(min_ret_20), 1e-6))
                - 0.35 * risk
            )
            if risk > 0.9:
                label = "no_trade"
                target = 0.0
                reason = "risk filter triggered"
                warning = "recent volatility or drawdown is extreme"
            elif prob5 >= strong_entry and prob20 >= entry and pred5 > min_ret_5 and pred20 > min_ret_20:
                label = "strong_positive"
                target = max_position
                reason = "5d and 20d expected return/probability both positive"
                warning = "model output is probabilistic, not a price forecast"
            elif (prob5 >= entry and pred5 > min_ret_5) or (prob20 >= entry and pred20 > min_ret_20):
                label = "positive"
                target = 0.5 * max_position
                reason = "one primary horizon has positive edge"
                warning = "confirm liquidity, limit status, and cost assumptions"
            elif prob5 <= exit_threshold and pred5 < 0:
                label = "negative"
                target = 0.0
                reason = "short horizon expected return is weak"
                warning = "risk/reward is unfavorable under current model"
            else:
                label = "neutral"
                target = float(cfg.get("neutral_position", 0.0))
                reason = "no clear model edge"
                warning = "observe only unless independent thesis exists"
        rows.append(
            {
                "date": row["date"],
                "close": row["close"],
                "pred_ret_5d": pred5 if h5 == 5 else row.get("pred_ret_5d", np.nan),
                "pred_ret_20d": pred20 if h20 == 20 else row.get("pred_ret_20d", np.nan),
                "prob_up_5d": prob5 if h5 == 5 else row.get("prob_up_5d", np.nan),
                "prob_up_20d": prob20 if h20 == 20 else row.get("prob_up_20d", np.nan),
                "risk_score": risk,
                "signal_score": score,
                "signal_label": label,
                "target_position": target,
                "reason": reason,
                "risk_warning": warning,
            }
        )
    return pd.DataFrame(rows)


def latest_signal(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    return signals.sort_values("date").tail(1).reset_index(drop=True)


def main() -> None:
    import argparse

    from .report import run_research_pipeline

    parser = argparse.ArgumentParser(description="Generate 300316.SZ daily model signal.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = run_research_pipeline(args.config)
    latest = latest_signal(result["signals"])
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()

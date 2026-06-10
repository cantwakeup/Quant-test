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


def _score_from_feature(value: float, scale: float, invert: bool = False) -> float:
    if pd.isna(value):
        return 0.5
    raw = 0.5 + float(value) / max(scale, 1e-9)
    score = _bounded(raw)
    return 1.0 - score if invert else score


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
        "ma_gap_60d",
        "trend_consistency_20d",
        "rsi_14d",
        "volume_ratio_20d",
        "amount_ratio_20d",
        "rs_chinext_20d",
        "rs_hs300_20d",
        "rs_industry_20d",
        "event_cooling_any_risk",
        "days_since_fin_report",
        "limit_up_20d_count",
        "limit_down_20d_count",
        "atr_pct_14d",
    ]
    data = data.merge(features[["date"] + [c for c in keep_features if c in features.columns]], on="date", how="left")

    max_position = float(cfg.get("max_position", 1.0))
    entry = float(cfg.get("entry_threshold", 0.56))
    strong_entry = float(cfg.get("strong_entry_threshold", 0.62))
    exit_threshold = float(cfg.get("exit_threshold", 0.48))
    min_ret_5 = float(cfg.get("min_expected_return_5d", 0.01))
    min_ret_20 = float(cfg.get("min_expected_return_20d", 0.025))
    event_cooling_days = int(cfg.get("event_cooling_days", 5))

    rows = []
    for _, row in data.iterrows():
        pred5 = row.get(f"pred_ret_{h5}d", np.nan)
        pred20 = row.get(f"pred_ret_{h20}d", np.nan)
        prob5 = row.get(f"prob_up_{h5}d", np.nan)
        prob20 = row.get(f"prob_up_{h20}d", np.nan)
        pred_excess5 = row.get("pred_excess_ret_5d", np.nan)
        risk = _risk_score(row, cfg)
        trend_consistency = row.get("trend_consistency_20d", np.nan)
        trend_consistency_score = 0.5 if pd.isna(trend_consistency) else _bounded(trend_consistency)
        trend = 0.55 * _score_from_feature(row.get("ma_gap_20d", np.nan), 0.08) + 0.45 * trend_consistency_score
        volume_score = 0.5 * _score_from_feature(row.get("volume_ratio_20d", np.nan) - 1.0, 1.5) + 0.5 * _score_from_feature(row.get("amount_ratio_20d", np.nan) - 1.0, 1.5)
        rs_candidates = [row.get(c, np.nan) for c in ("rs_chinext_20d", "rs_hs300_20d", "rs_industry_20d")]
        rs_values = [v for v in rs_candidates if pd.notna(v)]
        relative_strength = _score_from_feature(float(np.mean(rs_values)), 0.08) if rs_values else 0.5
        fundamental = 0.5
        if pd.notna(row.get("days_since_fin_report", np.nan)):
            fundamental = 0.6 if 0 <= row.get("days_since_fin_report", np.nan) <= 20 else 0.5
        event_risk = row.get("event_cooling_any_risk", 0.0)
        event_score = 1.0 - _bounded(event_risk) if pd.notna(event_risk) else 0.5
        near_limit = bool(row.get("limit_up", False)) or bool(row.get("limit_down", False))
        atr_pct = row.get("atr_pct_14d", np.nan)
        atr_pct = 0.04 if pd.isna(atr_pct) else float(atr_pct)
        stop_ref = row["close"] * (1.0 - max(atr_pct * 2.5, 0.08))
        take_ref = row["close"] * (1.0 + max(atr_pct * 3.0, 0.12))

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
                0.25 * ((prob5 - 0.5) / 0.15)
                + 0.18 * ((prob20 - 0.5) / 0.15)
                + 0.18 * (pred5 / max(abs(min_ret_5), 1e-6))
                + 0.14 * (pred20 / max(abs(min_ret_20), 1e-6))
                + 0.10 * (trend - 0.5) * 2.0
                + 0.06 * (volume_score - 0.5) * 2.0
                + 0.06 * (relative_strength - 0.5) * 2.0
                + 0.03 * (event_score - 0.5) * 2.0
                - 0.45 * risk
            )
            if near_limit:
                label = "no_trade"
                target = 0.0
                reason = "limit price or near-untradable state"
                warning = "execution may be unavailable or highly biased"
            elif event_score < 0.2 and event_cooling_days > 0:
                label = "no_trade"
                target = 0.0
                reason = "event cooling risk"
                warning = "event window may dominate model signal"
            elif risk > 0.9:
                label = "risk_off"
                target = 0.0
                reason = "risk filter triggered"
                warning = "recent volatility or drawdown is extreme"
            elif prob5 >= strong_entry and prob20 >= entry and pred5 > min_ret_5 and pred20 > min_ret_20:
                label = "strong_buy"
                target = max_position if risk < 0.55 else 0.75 * max_position
                reason = "5d and 20d expected return/probability both positive"
                warning = "model output is probabilistic, not a price forecast"
            elif (prob5 >= entry and pred5 > min_ret_5) or (prob20 >= entry and pred20 > min_ret_20):
                label = "buy"
                target = 0.5 * max_position if risk < 0.7 else 0.25 * max_position
                reason = "one primary horizon has positive edge"
                warning = "confirm liquidity, limit status, and cost assumptions"
            elif prob5 <= exit_threshold and pred5 < 0:
                label = "reduce"
                target = 0.0
                reason = "short horizon expected return is weak"
                warning = "risk/reward is unfavorable under current model"
            elif prob5 >= float(cfg.get("watch_threshold", 0.52)) or trend > 0.65:
                label = "watch"
                target = 0.25 * max_position if risk < 0.55 else 0.0
                reason = "partial evidence but not enough for full signal"
                warning = "observe; edge is not statistically strong"
            else:
                label = "no_trade"
                target = float(cfg.get("neutral_position", 0.0))
                reason = "no clear model edge"
                warning = "no statistically reliable advantage"
        rows.append(
            {
                "date": row["date"],
                "close": row["close"],
                "pred_ret_5d": pred5 if h5 == 5 else row.get("pred_ret_5d", np.nan),
                "pred_ret_20d": pred20 if h20 == 20 else row.get("pred_ret_20d", np.nan),
                "prob_up_5d": prob5 if h5 == 5 else row.get("prob_up_5d", np.nan),
                "prob_up_20d": prob20 if h20 == 20 else row.get("prob_up_20d", np.nan),
                "pred_excess_ret_5d": pred_excess5,
                "risk_score": risk,
                "trend_score": trend,
                "volume_score": volume_score,
                "relative_strength_score": relative_strength,
                "fundamental_score": fundamental,
                "event_score": event_score,
                "signal_score": score,
                "signal_label": label,
                "target_position": target,
                "stop_loss_reference": stop_ref,
                "take_profit_reference": take_ref,
                "invalidation_condition": "close below stop reference, risk_score above 0.9, or signal flips to reduce/risk_off",
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

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .utils import rank_corr


def feature_columns(features: pd.DataFrame) -> List[str]:
    blocked_prefixes = ("y_", "future_")
    blocked_names = {"date"}
    return [
        c
        for c in features.columns
        if c not in blocked_names
        and not c.startswith(blocked_prefixes)
        and pd.api.types.is_numeric_dtype(features[c])
    ]


def rolling_ic(factor: pd.Series, target: pd.Series, window: int = 252, method: str = "spearman") -> pd.Series:
    sample = pd.concat([factor, target], axis=1)
    if method == "spearman":
        sample = sample.rank()
    return sample.iloc[:, 0].rolling(window).corr(sample.iloc[:, 1])


def quantile_test(factor: pd.Series, target: pd.Series, quantiles: int = 5) -> pd.DataFrame:
    sample = pd.concat([factor, target], axis=1).dropna()
    sample.columns = ["factor", "target"]
    if len(sample) < quantiles * 5 or sample["factor"].nunique() < quantiles:
        return pd.DataFrame(columns=["quantile", "mean_return", "median_return", "count"])
    ranks = sample["factor"].rank(method="first")
    sample["quantile"] = pd.qcut(ranks, quantiles, labels=False, duplicates="drop") + 1
    return (
        sample.groupby("quantile")["target"]
        .agg(mean_return="mean", median_return="median", count="count")
        .reset_index()
    )


def yearly_ic(dates: pd.Series, factor: pd.Series, target: pd.Series) -> pd.DataFrame:
    sample = pd.DataFrame({"date": pd.to_datetime(dates), "factor": factor, "target": target}).dropna()
    if sample.empty:
        return pd.DataFrame(columns=["year", "ic", "rank_ic", "count"])
    rows = []
    for year, group in sample.groupby(sample["date"].dt.year):
        rows.append(
            {
                "year": int(year),
                "ic": rank_corr(group["factor"], group["target"], "pearson"),
                "rank_ic": rank_corr(group["factor"], group["target"], "spearman"),
                "count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def regime_ic(features: pd.DataFrame, factor: pd.Series, target: pd.Series) -> pd.DataFrame:
    rows = []
    candidates = {
        "high_vol": "volatility_20d",
        "strong_trend": "ma_gap_20d",
    }
    for name, col in candidates.items():
        if col not in features.columns:
            continue
        state = features[col]
        valid = pd.concat([state, factor, target], axis=1).dropna()
        if len(valid) < 80:
            continue
        threshold = valid.iloc[:, 0].median()
        for label, mask in (("low", valid.iloc[:, 0] <= threshold), ("high", valid.iloc[:, 0] > threshold)):
            part = valid.loc[mask]
            rows.append(
                {
                    "regime": f"{name}_{label}",
                    "rank_ic": rank_corr(part.iloc[:, 1], part.iloc[:, 2], "spearman"),
                    "count": int(len(part)),
                }
            )
    return pd.DataFrame(rows)


def monotonicity_score(qtest: pd.DataFrame) -> float:
    if qtest is None or qtest.empty or len(qtest) < 3:
        return 0.0
    score = abs(rank_corr(qtest["quantile"], qtest["mean_return"], "spearman"))
    return 0.0 if pd.isna(score) else float(score)


def low_turnover_score(series: pd.Series, bins: int = 5) -> float:
    sample = series.dropna()
    if len(sample) < 20 or sample.nunique() <= 1:
        return 0.5
    ranks = sample.rank(method="first")
    q = pd.qcut(ranks, bins, labels=False, duplicates="drop")
    changes = pd.Series(q, index=sample.index).diff().abs()
    avg_change = changes.dropna().mean()
    if pd.isna(avg_change):
        return 0.5
    return float(1.0 / (1.0 + avg_change))


def explainability_score(name: str) -> float:
    readable_prefixes = (
        "ret_",
        "ma_",
        "break_",
        "dist_",
        "macd",
        "rsi",
        "kdj",
        "boll",
        "volume",
        "amount",
        "turnover",
        "volatility",
        "downside",
        "max_drawdown",
        "atr",
        "intraday",
        "rs_",
        "fund_",
        "post_",
        "days_since",
    )
    return 0.9 if name.startswith(readable_prefixes) else 0.7


def evaluate_factor_pool(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizon: int = 5,
    min_obs: int = 180,
    rolling_window: int = 252,
    quantiles: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = features.merge(labels[["date", f"y_ret_{horizon}d"]], on="date", how="inner")
    target = merged[f"y_ret_{horizon}d"]
    rows = []
    q_rows = []
    y_rows = []
    r_rows = []
    cols = [c for c in feature_columns(features) if c in merged.columns]
    corr_matrix = merged[cols].corr(method="spearman") if cols else pd.DataFrame()
    redundancy = {}
    if not corr_matrix.empty:
        for col in cols:
            others = corr_matrix[col].drop(index=col, errors="ignore").abs()
            redundancy[col] = float(others.max()) if not others.empty else 0.0

    for col in cols:
        sample = pd.concat([merged[col], target], axis=1).dropna()
        sample.columns = ["factor", "target"]
        if len(sample) < min_obs or sample["factor"].nunique() <= 2:
            continue
        pearson = rank_corr(sample["factor"], sample["target"], "pearson")
        spearman = rank_corr(sample["factor"], sample["target"], "spearman")
        ric = rolling_ic(merged[col], target, rolling_window, method="spearman")
        ric_mean = float(ric.mean()) if ric.notna().any() else np.nan
        ric_std = float(ric.std(ddof=0)) if ric.notna().any() else np.nan
        icir = ric_mean / ric_std if ric_std and np.isfinite(ric_std) else np.nan

        qtest = quantile_test(merged[col], target, quantiles=quantiles)
        for _, qrow in qtest.iterrows():
            q_rows.append({"factor": col, **qrow.to_dict()})
        mono = monotonicity_score(qtest)

        yic = yearly_ic(merged["date"], merged[col], target)
        stable_sign = 0.0
        if not yic.empty and yic["rank_ic"].notna().any():
            signs = np.sign(yic["rank_ic"].dropna())
            major = np.sign(spearman) if pd.notna(spearman) and spearman != 0 else 0
            stable_sign = float((signs == major).mean()) if major != 0 else 0.0
        for _, yrow in yic.iterrows():
            y_rows.append({"factor": col, **yrow.to_dict()})

        reg = regime_ic(merged, merged[col], target)
        for _, rrow in reg.iterrows():
            r_rows.append({"factor": col, **rrow.to_dict()})

        pred_score = min(abs(spearman) / 0.08, 1.0) if pd.notna(spearman) else 0.0
        icir_score = min(abs(icir) / 0.5, 1.0) if pd.notna(icir) else 0.0
        stability = 0.5 * stable_sign + 0.5 * icir_score
        non_redundancy = 1.0 - min(redundancy.get(col, 0.0), 1.0)
        low_turnover = low_turnover_score(merged[col], bins=quantiles)
        interpret = explainability_score(col)
        total = (
            0.30 * pred_score
            + 0.20 * stability
            + 0.15 * mono
            + 0.15 * non_redundancy
            + 0.10 * low_turnover
            + 0.10 * interpret
        )
        rows.append(
            {
                "factor": col,
                "horizon": horizon,
                "observations": int(len(sample)),
                "pearson_ic": pearson,
                "spearman_ic": spearman,
                "rolling_rank_ic_mean": ric_mean,
                "rolling_rank_ic_std": ric_std,
                "icir": icir,
                "prediction_score": pred_score,
                "stability_score": stability,
                "monotonicity_score": mono,
                "non_redundancy_score": non_redundancy,
                "low_turnover_score": low_turnover,
                "explainability_score": interpret,
                "factor_score": total,
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("factor_score", ascending=False).reset_index(drop=True)
    return summary, pd.DataFrame(q_rows), pd.DataFrame(y_rows), pd.DataFrame(r_rows)


def factor_correlation_report(features: pd.DataFrame, selected: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in selected if c in features.columns]
    if not cols:
        return pd.DataFrame()
    corr = features[cols].corr(method="spearman").abs()
    rows = []
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            rows.append({"factor_a": left, "factor_b": right, "abs_corr": corr.loc[left, right]})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .utils import rank_corr


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    blocked_prefixes = ("y_", "future_")
    return [
        c
        for c in df.columns
        if c != "date"
        and not c.startswith(blocked_prefixes)
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def score_features_on_train(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    min_non_null: int = 252,
) -> pd.DataFrame:
    rows = []
    for col in numeric_feature_columns(x_train):
        sample = pd.concat([x_train[col], y_train], axis=1).dropna()
        sample.columns = ["feature", "target"]
        if len(sample) < min_non_null or sample["feature"].nunique() <= 2:
            continue
        ic = rank_corr(sample["feature"], sample["target"], "pearson")
        rank_ic = rank_corr(sample["feature"], sample["target"], "spearman")
        rows.append(
            {
                "feature": col,
                "train_observations": int(len(sample)),
                "train_ic": ic,
                "train_rank_ic": rank_ic,
                "abs_train_rank_ic": abs(rank_ic) if pd.notna(rank_ic) else 0.0,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("abs_train_rank_ic", ascending=False).reset_index(drop=True)


def remove_correlated_features(
    x_train: pd.DataFrame,
    ranked_features: List[str],
    max_corr: float = 0.9,
) -> List[str]:
    selected: List[str] = []
    if not ranked_features:
        return selected
    corr = x_train[ranked_features].corr(method="spearman").abs()
    for feature in ranked_features:
        if feature not in corr.columns:
            continue
        if not selected:
            selected.append(feature)
            continue
        too_close = corr.loc[feature, selected].max(skipna=True)
        if pd.isna(too_close) or too_close < max_corr:
            selected.append(feature)
    return selected


def select_features_train_only(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 40,
    max_corr: float = 0.9,
    min_non_null: int = 252,
) -> Tuple[List[str], pd.DataFrame]:
    scores = score_features_on_train(x_train, y_train, min_non_null=min_non_null)
    if scores.empty:
        fallback = numeric_feature_columns(x_train)
        fallback = [c for c in fallback if x_train[c].notna().sum() >= max(20, min_non_null // 4)]
        return fallback[:top_n], scores
    ranked = scores["feature"].head(max(top_n * 3, top_n)).tolist()
    selected = remove_correlated_features(x_train, ranked, max_corr=max_corr)
    return selected[:top_n], scores

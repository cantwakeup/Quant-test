from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd

from .feature_selection import numeric_feature_columns, select_features_train_only
from .models import NumpyLogisticClassifier, NumpyRidgeRegressor, evaluate_prediction_frame, try_import_sklearn


def purged_walk_forward_splits(
    n_rows: int,
    min_train_size: int = 756,
    test_size: int = 63,
    purge: int = 20,
    embargo: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    test_start = int(min_train_size)
    while test_start < n_rows:
        train_end = max(0, test_start - int(purge))
        test_end = min(n_rows, test_start + int(test_size))
        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx
        test_start = test_end + int(embargo)


def run_walk_forward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
    walk_config: Dict[str, object] | None = None,
    model_config: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    walk_config = walk_config or {}
    model_config = model_config or {}
    merged = features.merge(labels, on="date", how="left").sort_values("date").reset_index(drop=True)
    feature_cols = numeric_feature_columns(features)
    predictions = pd.DataFrame({"date": merged["date"]})
    for horizon in horizons:
        predictions[f"pred_ret_{horizon}d"] = np.nan
        predictions[f"prob_up_{horizon}d"] = np.nan

    diagnostics: List[Dict[str, object]] = []
    sklearn_status = try_import_sklearn() if model_config.get("use_sklearn_if_available", False) else {"available": False, "error": "disabled"}

    for split_id, (train_idx, test_idx) in enumerate(
        purged_walk_forward_splits(
            len(merged),
            min_train_size=int(walk_config.get("min_train_size", 756)),
            test_size=int(walk_config.get("test_size", 63)),
            purge=int(walk_config.get("purge", 20)),
            embargo=int(walk_config.get("embargo", 0)),
        ),
        start=1,
    ):
        x_train_all = merged.loc[train_idx, feature_cols]
        x_test_all = merged.loc[test_idx, feature_cols]
        for horizon in horizons:
            y_reg_col = f"y_ret_{horizon}d"
            y_cls_col = f"y_up_{horizon}d"
            if y_reg_col not in merged.columns or y_cls_col not in merged.columns:
                continue
            y_train = merged.loc[train_idx, y_reg_col]
            selected, scores = select_features_train_only(
                x_train_all,
                y_train,
                top_n=int(walk_config.get("top_n_features", 40)),
                max_corr=float(walk_config.get("max_feature_corr", 0.9)),
                min_non_null=int(walk_config.get("min_train_non_null", 252)),
            )
            x_train = x_train_all[selected]
            x_test = x_test_all[selected]

            ridge = NumpyRidgeRegressor(alpha=float(model_config.get("ridge_alpha", 10.0)))
            ridge.fit(x_train, y_train)
            predictions.loc[test_idx, f"pred_ret_{horizon}d"] = ridge.predict(x_test)

            clf = NumpyLogisticClassifier(
                l2=float(model_config.get("logistic_l2", 1.0)),
                learning_rate=float(model_config.get("logistic_lr", 0.05)),
                n_iter=int(model_config.get("logistic_iter", 700)),
            )
            clf.fit(x_train, merged.loc[train_idx, y_cls_col])
            predictions.loc[test_idx, f"prob_up_{horizon}d"] = clf.predict_proba(x_test)

            diagnostics.append(
                {
                    "split": split_id,
                    "horizon": int(horizon),
                    "train_start": merged.loc[train_idx[0], "date"],
                    "train_end": merged.loc[train_idx[-1], "date"],
                    "test_start": merged.loc[test_idx[0], "date"],
                    "test_end": merged.loc[test_idx[-1], "date"],
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "selected_feature_count": int(len(selected)),
                    "selected_features": ",".join(selected[:20]),
                    "top_train_feature": scores.iloc[0]["feature"] if not scores.empty else "",
                    "sklearn_available": bool(sklearn_status.get("available", False)),
                    "sklearn_error": sklearn_status.get("error", ""),
                }
            )

    metric_rows = []
    for horizon in horizons:
        metric = evaluate_prediction_frame(predictions, labels, int(horizon))
        metric_rows.append(metric.__dict__)
    metrics = pd.DataFrame(metric_rows)
    return predictions, pd.DataFrame(diagnostics), metrics

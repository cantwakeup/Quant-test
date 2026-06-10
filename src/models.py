from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


class MedianStandardizer:
    def __init__(self):
        self.median_: Optional[pd.Series] = None
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None
        self.columns_: Optional[list] = None

    def fit(self, x: pd.DataFrame) -> "MedianStandardizer":
        self.columns_ = list(x.columns)
        numeric = x.astype(float)
        self.median_ = numeric.median()
        filled = numeric.fillna(self.median_)
        self.mean_ = filled.mean()
        self.std_ = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if self.columns_ is None or self.median_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer has not been fit.")
        aligned = x.reindex(columns=self.columns_).astype(float)
        filled = aligned.fillna(self.median_)
        z = (filled - self.mean_) / self.std_
        return z.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=float)


class NumpyRidgeRegressor:
    def __init__(self, alpha: float = 10.0):
        self.alpha = float(alpha)
        self.scaler = MedianStandardizer()
        self.coef_: Optional[np.ndarray] = None
        self.y_mean_: float = 0.0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "NumpyRidgeRegressor":
        sample = pd.concat([x, y.rename("_target")], axis=1).dropna(subset=["_target"])
        if sample.empty or x.shape[1] == 0:
            self.y_mean_ = float(pd.Series(y).mean()) if pd.Series(y).notna().any() else 0.0
            self.coef_ = None
            return self
        x_sample = sample.drop(columns=["_target"])
        y_sample = sample["_target"].astype(float)
        self.y_mean_ = float(y_sample.mean())
        xz = self.scaler.fit(x_sample).transform(x_sample)
        design = np.column_stack([np.ones(len(xz)), xz])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        try:
            self.coef_ = np.linalg.solve(design.T @ design + penalty, design.T @ y_sample.to_numpy())
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_sample.to_numpy()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            return np.full(len(x), self.y_mean_, dtype=float)
        xz = self.scaler.transform(x)
        design = np.column_stack([np.ones(len(xz)), xz])
        return design @ self.coef_


class NumpyLogisticClassifier:
    def __init__(self, l2: float = 1.0, learning_rate: float = 0.05, n_iter: int = 700):
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.n_iter = int(n_iter)
        self.scaler = MedianStandardizer()
        self.coef_: Optional[np.ndarray] = None
        self.prior_: float = 0.5

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -35, 35)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "NumpyLogisticClassifier":
        sample = pd.concat([x, y.rename("_target")], axis=1).dropna(subset=["_target"])
        if sample.empty or x.shape[1] == 0:
            self.prior_ = float(pd.Series(y).mean()) if pd.Series(y).notna().any() else 0.5
            self.coef_ = None
            return self
        y_sample = sample["_target"].astype(float).clip(0, 1)
        self.prior_ = float(y_sample.mean())
        if y_sample.nunique() < 2:
            self.coef_ = None
            return self
        x_sample = sample.drop(columns=["_target"])
        xz = self.scaler.fit(x_sample).transform(x_sample)
        design = np.column_stack([np.ones(len(xz)), xz])
        beta = np.zeros(design.shape[1], dtype=float)
        beta[0] = np.log(self.prior_ / (1.0 - self.prior_))
        for _ in range(self.n_iter):
            pred = self._sigmoid(design @ beta)
            grad = design.T @ (pred - y_sample.to_numpy()) / len(y_sample)
            grad[1:] += self.l2 * beta[1:] / len(y_sample)
            beta -= self.learning_rate * grad
        self.coef_ = beta
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            prior = np.clip(self.prior_, 0.001, 0.999)
            return np.full(len(x), prior, dtype=float)
        xz = self.scaler.transform(x)
        design = np.column_stack([np.ones(len(xz)), xz])
        return self._sigmoid(design @ self.coef_)


def try_import_sklearn() -> Dict[str, object]:
    """Try importing sklearn lazily.

    Some restricted sandboxes break scipy temp-file probing. The system remains
    usable with the numpy models above when this import fails.
    """

    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
        from sklearn.inspection import permutation_importance  # type: ignore
        from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        return {
            "available": True,
            "RandomForestRegressor": RandomForestRegressor,
            "permutation_importance": permutation_importance,
            "ElasticNet": ElasticNet,
            "Lasso": Lasso,
            "LogisticRegression": LogisticRegression,
            "Ridge": Ridge,
            "Pipeline": Pipeline,
            "StandardScaler": StandardScaler,
            "error": "",
        }
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


@dataclass
class PredictionMetrics:
    horizon: int
    observations: int
    pearson_ic: float
    spearman_ic: float
    rmse: float
    direction_accuracy: float
    prob_accuracy: float


def evaluate_prediction_frame(predictions: pd.DataFrame, labels: pd.DataFrame, horizon: int) -> PredictionMetrics:
    merged = predictions.merge(labels, on="date", how="inner")
    reg_col = f"pred_ret_{horizon}d"
    prob_col = f"prob_up_{horizon}d"
    y_col = f"y_ret_{horizon}d"
    up_col = f"y_up_{horizon}d"
    sample = merged[[reg_col, prob_col, y_col, up_col]].dropna(subset=[reg_col, y_col])
    if sample.empty:
        return PredictionMetrics(horizon, 0, np.nan, np.nan, np.nan, np.nan, np.nan)
    pearson = sample[reg_col].corr(sample[y_col], method="pearson")
    spearman = sample[reg_col].rank().corr(sample[y_col].rank())
    rmse = float(np.sqrt(np.mean((sample[reg_col] - sample[y_col]) ** 2)))
    direction_accuracy = float(((sample[reg_col] > 0) == (sample[y_col] > 0)).mean())
    prob_sample = sample.dropna(subset=[prob_col, up_col])
    prob_accuracy = float(((prob_sample[prob_col] >= 0.5) == (prob_sample[up_col] > 0.5)).mean()) if len(prob_sample) else np.nan
    return PredictionMetrics(
        horizon=int(horizon),
        observations=int(len(sample)),
        pearson_ic=float(pearson) if pd.notna(pearson) else np.nan,
        spearman_ic=float(spearman) if pd.notna(spearman) else np.nan,
        rmse=rmse,
        direction_accuracy=direction_accuracy,
        prob_accuracy=prob_accuracy,
    )

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is in requirements.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Optional[str], base_dir: Optional[Path] = None) -> Optional[Path]:
    if path in (None, ""):
        return None
    candidate = Path(str(path))
    if candidate.is_absolute():
        return candidate
    return (base_dir or PROJECT_ROOT).joinpath(candidate).resolve()


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = resolve_path(path or "config.yaml", PROJECT_ROOT)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def ensure_directories(config: Dict[str, Any]) -> None:
    report_cfg = config.get("reports", {})
    base = Path(config.get("_config_dir", PROJECT_ROOT))
    for key in ("output_dir", "processed_dir"):
        path = resolve_path(report_cfg.get(key), base)
        if path:
            path.mkdir(parents=True, exist_ok=True)
    for rel in ("data/raw", "data/processed", "reports", "notebooks", "tests"):
        (base / rel).mkdir(parents=True, exist_ok=True)


def safe_divide(numerator: Any, denominator: Any) -> Any:
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def clean_numeric_frame(df: pd.DataFrame, exclude: Iterable[str] = ("date",)) -> pd.DataFrame:
    result = df.copy()
    excluded = set(exclude)
    for column in result.columns:
        if column not in excluded:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def rank_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    sample = pd.concat([x, y], axis=1).dropna()
    if len(sample) < 3:
        return np.nan
    if sample.iloc[:, 0].nunique() <= 1 or sample.iloc[:, 1].nunique() <= 1:
        return np.nan
    if method == "pearson":
        return float(sample.iloc[:, 0].corr(sample.iloc[:, 1], method="pearson"))
    return float(sample.iloc[:, 0].rank().corr(sample.iloc[:, 1].rank()))


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values.astype(float)
        y = y - y.mean()
        level = np.nanmean(np.abs(values))
        if not np.isfinite(level) or level == 0:
            return np.nan
        return float(np.dot(x, y) / denom / level)

    return series.rolling(window).apply(_slope, raw=True)


def rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    def _mdd(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        running_max = np.maximum.accumulate(values)
        drawdown = values / running_max - 1.0
        return float(np.min(drawdown))

    return close.rolling(window).apply(_mdd, raw=True)


def max_drawdown(equity: pd.Series) -> float:
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return np.nan
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())


def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    equity = pd.Series(equity).dropna()
    if len(equity) < 2:
        return np.nan
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / periods_per_year
    if years <= 0 or equity.iloc[0] <= 0:
        return np.nan
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def performance_metrics(
    returns: pd.Series,
    equity: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    position: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    returns = pd.Series(returns).fillna(0.0)
    if equity is None:
        equity = (1.0 + returns).cumprod()
    equity = pd.Series(equity).replace([np.inf, -np.inf], np.nan).dropna()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else np.nan
    ann_ret = annualized_return(equity, periods_per_year)
    ann_vol = float(returns.std(ddof=0) * math.sqrt(periods_per_year)) if len(returns) else np.nan
    sharpe = float(ann_ret / ann_vol) if ann_vol and np.isfinite(ann_vol) else np.nan
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0) * math.sqrt(periods_per_year)) if len(downside) else np.nan
    sortino = float(ann_ret / downside_vol) if downside_vol and np.isfinite(downside_vol) else np.nan
    mdd = max_drawdown(equity)
    calmar = float(ann_ret / abs(mdd)) if mdd and np.isfinite(mdd) and mdd < 0 else np.nan

    trade_count = np.nan
    avg_holding_days = np.nan
    turnover = np.nan
    if position is not None:
        pos = pd.Series(position).fillna(0.0)
        changes = pos.diff().abs().fillna(pos.abs())
        trade_count = float((changes > 1e-12).sum())
        turnover = float(changes.sum())
        groups = (pos.ne(pos.shift())).cumsum()
        holding_lengths = pos.groupby(groups).agg(["first", "size"])
        active_lengths = holding_lengths.loc[holding_lengths["first"] > 0, "size"]
        avg_holding_days = float(active_lengths.mean()) if len(active_lengths) else 0.0

    active = returns[returns != 0]
    wins = active[active > 0]
    losses = active[active < 0]
    win_rate = float(len(wins) / len(active)) if len(active) else np.nan
    profit_loss = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan
    losing_streak = 0
    max_losing_streak = 0
    for value in active:
        if value < 0:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        else:
            losing_streak = 0

    metrics = {
        "total_return": total_return,
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss,
        "trade_count": trade_count,
        "avg_holding_days": avg_holding_days,
        "turnover": turnover,
        "single_period_max_loss": float(returns.min()) if len(returns) else np.nan,
        "max_consecutive_losses": float(max_losing_streak),
    }
    if benchmark_returns is not None:
        aligned = pd.concat([returns, pd.Series(benchmark_returns)], axis=1).fillna(0.0)
        excess_equity = (1.0 + aligned.iloc[:, 0] - aligned.iloc[:, 1]).cumprod()
        metrics["excess_total_return"] = float(excess_equity.iloc[-1] / excess_equity.iloc[0] - 1.0)
    return metrics


def to_markdown_table(df: pd.DataFrame, float_format: str = ".4f", max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No data available._"
    view = df.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: "" if pd.isna(x) else format(float(x), float_format))
    return view.to_markdown(index=False)


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

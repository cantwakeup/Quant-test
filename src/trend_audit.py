from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import annual_return_table, extract_trade_log, regime_return_table, run_vector_backtest
from .null_tests import null_signal_test
from .strategy_rules import StrategySpec, build_signal_from_spec
from .utils import to_markdown_table, write_markdown


MA_FAST = [5, 10, 20, 30, 60]
MA_SLOW = [60, 90, 120, 180, 250]
MOM_WINDOWS = [10, 20, 40, 60, 120]
RISK_FILTERS = ["none", "close_gt_ma120", "close_gt_ma250", "vol_pct_lt_80", "mdd60_gt_-25", "amount_liquidity", "gap_risk", "limit_tradable"]


def _evaluate_spec(prices: pd.DataFrame, spec: StrategySpec, config: Dict[str, object]) -> Dict[str, object]:
    signal = build_signal_from_spec(prices, spec)
    bt, metrics = run_vector_backtest(prices, signal, config=config)
    strat = metrics.loc[metrics["portfolio"] == "strategy"].iloc[0].to_dict()
    buy = metrics.loc[metrics["portfolio"] == "buy_hold_300316"].iloc[0].to_dict()
    return {
        **strat,
        "strategy_name": spec.name,
        "kind": spec.kind,
        "fast": spec.fast,
        "slow": spec.slow,
        "window": spec.window,
        "risk_filter": spec.risk_filter,
        "buy_hold_total_return": buy["total_return"],
        "beats_buy_hold": strat["total_return"] > buy["total_return"],
    }


def parameter_grid(prices: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for risk_filter in RISK_FILTERS:
        for fast in MA_FAST:
            for slow in MA_SLOW:
                if fast >= slow:
                    continue
                rows.append(_evaluate_spec(prices, StrategySpec(f"ma_{fast}_{slow}_{risk_filter}", "ma", fast=fast, slow=slow, risk_filter=risk_filter), config))
        for window in MOM_WINDOWS:
            rows.append(_evaluate_spec(prices, StrategySpec(f"momentum_{window}_{risk_filter}", "momentum", window=window, risk_filter=risk_filter), config))
    return pd.DataFrame(rows).sort_values(["beats_buy_hold", "calmar", "total_return"], ascending=[False, False, False])


def walk_forward_parameter_selection(
    prices: pd.DataFrame,
    config: Dict[str, object],
    train_years: int = 3,
    mode: str = "anchored",
) -> pd.DataFrame:
    dates = pd.to_datetime(prices["date"])
    years = sorted(dates.dt.year.unique())
    rows = []
    for test_year in years:
        if test_year < years[0] + train_years:
            continue
        if mode == "anchored":
            train_mask = dates.dt.year < test_year
        else:
            train_mask = (dates.dt.year < test_year) & (dates.dt.year >= test_year - train_years)
        test_mask = dates.dt.year == test_year
        train_prices = prices.loc[train_mask].reset_index(drop=True)
        test_prices = prices.loc[test_mask].reset_index(drop=True)
        if len(train_prices) < 252 or len(test_prices) < 40:
            continue
        grid = parameter_grid(train_prices, config)
        if grid.empty:
            continue
        best = grid.iloc[0]
        spec = StrategySpec(
            str(best["strategy_name"]),
            str(best["kind"]),
            fast=int(best["fast"]) if pd.notna(best.get("fast")) else None,
            slow=int(best["slow"]) if pd.notna(best.get("slow")) else None,
            window=int(best["window"]) if pd.notna(best.get("window")) else None,
            risk_filter=str(best["risk_filter"]),
        )
        out = _evaluate_spec(test_prices, spec, config)
        out["selection_mode"] = mode
        out["test_year"] = int(test_year)
        out["selected_on_train_total_return"] = best["total_return"]
        rows.append(out)
    return pd.DataFrame(rows)


def _plot_heatmap(grid: pd.DataFrame, output_path: Path) -> None:
    ma = grid[(grid["kind"] == "ma") & (grid["risk_filter"] == "none")].copy()
    if ma.empty:
        return
    pivot = ma.pivot_table(index="fast", columns="slow", values="total_return")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), labels=[str(i) for i in pivot.index])
    ax.set_xlabel("slow MA")
    ax.set_ylabel("fast MA")
    ax.set_title("MA grid total return, no risk filter")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def run_trend_audit(prices: pd.DataFrame, config: Dict[str, object], report_dir: Path, processed_dir: Path) -> Dict[str, pd.DataFrame]:
    grid = parameter_grid(prices, config)
    grid.to_csv(report_dir / "trend_parameter_grid.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(processed_dir / "trend_parameter_grid.csv", index=False, encoding="utf-8-sig")
    _plot_heatmap(grid, report_dir / "trend_parameter_heatmap.png")

    focus_specs = [
        StrategySpec("ma_60_120", "ma", fast=60, slow=120),
        StrategySpec("momentum_20", "momentum", window=20),
    ]
    focus_rows = []
    all_trade_logs = []
    null_rows = []
    for spec in focus_specs:
        signal = build_signal_from_spec(prices, spec)
        bt, metrics = run_vector_backtest(prices, signal, config=config)
        trade_log = extract_trade_log(bt)
        if not trade_log.empty:
            trade_log.insert(0, "strategy_name", spec.name)
            all_trade_logs.append(trade_log)
        strat = metrics[metrics["portfolio"] == "strategy"].iloc[0].to_dict()
        focus_rows.append({"strategy_name": spec.name, **strat})
        null_rows.append(null_signal_test(prices, signal, strat["total_return"], config, n_iter=100, seed=2026).assign(strategy_name=spec.name))
    focus = pd.DataFrame(focus_rows)
    trend_trade_log = pd.concat(all_trade_logs, ignore_index=True) if all_trade_logs else pd.DataFrame()
    null_result = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    anchored = walk_forward_parameter_selection(prices, config, mode="anchored")
    rolling = walk_forward_parameter_selection(prices, config, mode="rolling")
    annual = annual_return_table(run_vector_backtest(prices, build_signal_from_spec(prices, focus_specs[0]), config=config)[0])
    regime = regime_return_table(run_vector_backtest(prices, build_signal_from_spec(prices, focus_specs[0]), config=config)[0], prices)

    trend_trade_log.to_csv(report_dir / "trend_trade_log.csv", index=False, encoding="utf-8-sig")
    null_result.to_csv(processed_dir / "trend_null_tests.csv", index=False, encoding="utf-8-sig")
    anchored.to_csv(processed_dir / "trend_walk_forward_anchored.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(processed_dir / "trend_walk_forward_rolling.csv", index=False, encoding="utf-8-sig")

    pvals = []
    for strategy_name, group in null_result.groupby("strategy_name") if not null_result.empty else []:
        pvals.append({"strategy_name": strategy_name, "empirical_p_value": group["beats_base"].mean(), "null_median_return": group["total_return"].median()})
    pval_df = pd.DataFrame(pvals)

    top5_contrib = pd.DataFrame()
    if not trend_trade_log.empty:
        tmp = trend_trade_log.sort_values("trade_return", ascending=False)
        top5_contrib = tmp.groupby("strategy_name").head(5).groupby("strategy_name")["trade_return"].sum().reset_index(name="top5_trade_return_sum")

    content = f"""# 趋势策略审计报告

## 规则定义

- `ma_60_120`：收盘后判断 60 日均线是否高于 120 日均线；下一交易日开盘成交。
- `momentum_20`：收盘后判断过去 20 日收益是否大于 0；下一交易日开盘成交。
- 成本使用配置中的 commission、slippage、stamp tax、minimum fee。
- 日频近似 A 股 T+1，不允许同日信号同日成交；涨跌停/停牌字段可用时限制成交。

## 重点策略结果

{to_markdown_table(focus)}

## 参数网格 Top 20

{to_markdown_table(grid.head(20))}

## Walk-forward 参数选择

Anchored:

{to_markdown_table(anchored)}

Rolling:

{to_markdown_table(rolling)}

## 收益来源集中度

{to_markdown_table(top5_contrib)}

## 分年与分状态

{to_markdown_table(annual)}

{to_markdown_table(regime)}

## 随机信号反证

经验 p-value 表示随机信号收益大于等于基准趋势策略的比例，越小越支持趋势策略不是随机偶然。

{to_markdown_table(pval_df)}

## 结论

趋势基线显示“有条件有效”的线索，但不能直接升级为可交易：需要通过 walk-forward 参数选择、延迟成交、成本扰动和 2022 年后样本继续验证。若优势主要来自少数交易或 2020-2021 大行情，应降级为仅供观察。
"""
    write_markdown(report_dir / "trend_strategy_audit.md", content)
    return {
        "trend_parameter_grid": grid,
        "trend_focus_metrics": focus,
        "trend_trade_log": trend_trade_log,
        "trend_null_tests": null_result,
        "trend_walk_forward_anchored": anchored,
        "trend_walk_forward_rolling": rolling,
    }

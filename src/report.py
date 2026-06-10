from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import run_baseline_backtests, run_vector_backtest
from .data_adapter import build_adapter
from .data_cleaning import make_peer_index
from .factor_analysis import evaluate_factor_pool, factor_correlation_report
from .feature_engineering import build_feature_table
from .label_builder import build_labels
from .signal_generator import generate_signals, latest_signal
from .utils import ensure_directories, load_config, resolve_path, to_markdown_table, write_markdown
from .walk_forward import run_walk_forward


def _first_available_index(indices: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    for _, df in indices.items():
        if df is not None and not df.empty:
            return df
    return None


def _classification(backtest_metrics: pd.DataFrame, model_metrics: pd.DataFrame) -> str:
    if backtest_metrics.empty or model_metrics.empty:
        return "C. 不可用"
    strat = backtest_metrics.loc[backtest_metrics["portfolio"] == "strategy"]
    buy = backtest_metrics.loc[backtest_metrics["portfolio"] == "buy_hold_300316"]
    if strat.empty or buy.empty:
        return "C. 不可用"
    s = strat.iloc[0]
    b = buy.iloc[0]
    primary = model_metrics[model_metrics["horizon"].isin([5, 20])]
    avg_rank_ic = primary["spearman_ic"].mean() if not primary.empty else np.nan
    enough_trades = s.get("trade_count", 0) >= 10
    cost_advantage = s.get("total_return", -999) > b.get("total_return", 999)
    controlled_dd = s.get("max_drawdown", -1) > -0.35
    if cost_advantage and controlled_dd and enough_trades and pd.notna(avg_rank_ic) and avg_rank_ic > 0.03:
        return "A. 可交易"
    if pd.notna(avg_rank_ic) and avg_rank_ic > 0 and s.get("max_drawdown", -1) > -0.50:
        return "B. 仅供观察"
    return "C. 不可用"


def _write_equity_plot(backtest: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(backtest["date"], backtest["strategy_equity"], label="strategy")
    axes[0].plot(backtest["date"], backtest["buy_hold_equity"], label="buy_hold_300316")
    if backtest["benchmark_equity"].notna().any():
        axes[0].plot(backtest["date"], backtest["benchmark_equity"], label="benchmark")
    axes[0].legend()
    axes[0].set_title("Equity Curve")
    for col, label in (("strategy_equity", "strategy"), ("buy_hold_equity", "buy_hold")):
        equity = backtest[col]
        dd = equity / equity.cummax() - 1.0
        axes[1].plot(backtest["date"], dd, label=label)
    axes[1].legend()
    axes[1].set_title("Drawdown")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _write_factor_report(
    path: Path,
    config: Dict[str, object],
    factor_summary: pd.DataFrame,
    quantiles: pd.DataFrame,
    yearly: pd.DataFrame,
    regimes: pd.DataFrame,
    data_notes: str,
) -> None:
    top = factor_summary.head(20)
    content = f"""
# 300316.SZ 因子有效性报告

## 数据说明

{data_notes}

本报告使用单股票时间序列检验。它可以回答“该股票自身历史状态是否与未来收益有关”，不能替代多股票横截面 IC 研究。

## 方法

- 标签：未来 1/3/5/10/20 个交易日 log return，本报告主展示 5 日。
- 因子：趋势/动量、反转、成交量、波动率、相对强弱、市场状态、财务公告日对齐特征、事件公告日特征。
- 评分：预测力、稳定性、单调性、非冗余性、低换手、可解释性加权。
- 注意：因子评分用于研究排序，不作为全样本选因子后直接回测；模型训练阶段会在每个 walk-forward 训练窗口内重新选特征。

## Top 因子评分

{to_markdown_table(top)}

## 分组检验样例

{to_markdown_table(quantiles.head(30))}

## 年度稳定性样例

{to_markdown_table(yearly.head(30))}

## 状态分段样例

{to_markdown_table(regimes.head(30))}

## 结论

因子报告只给出候选因子证据。若某些因子仅在少数年份或少数状态有效，应降低权重；若高度相关，则优先保留经济含义更清晰、换手更低的因子。
"""
    write_markdown(path, content)


def _write_model_report(
    path: Path,
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    conclusion: str,
) -> None:
    content = f"""
# 300316.SZ 模型训练报告

## Walk-forward 设计

- 训练集采用 expanding window。
- 测试集按时间顺序滚动。
- purge 默认 20 个交易日，用于降低重叠标签的信息泄漏。
- 缺失值填充、标准化、特征筛选都在每个训练窗口内 fit，再应用到对应测试窗口。
- 当前环境若无法导入 scikit-learn，会使用 numpy 实现的 Ridge/Logistic 基线，报告中保留 sklearn 可用性诊断。

## 样本外预测指标

{to_markdown_table(model_metrics)}

## Walk-forward 诊断样例

{to_markdown_table(diagnostics.head(20))}

## 研究结论

当前自动判定：**{conclusion}**。

这个结论来自样本外预测和扣成本回测的组合，不以单一 IC 或单次收益曲线为准。
"""
    write_markdown(path, content)


def _write_backtest_report(
    path: Path,
    backtest_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    latest: pd.DataFrame,
    conclusion: str,
    data_notes: str,
) -> None:
    content = f"""
# 300316.SZ 回测报告

## 执行假设

- 信号在收盘后生成。
- 默认下一交易日开盘成交。
- 成本包含佣金、印花税和滑点。
- 若行情数据含停牌/涨跌停字段则用于限制交易；本地 CSV 仅能用成交量和涨跌幅近似识别。
- A 股 T+1 约束以日频隔夜持仓近似处理，未模拟盘中撤单、排队和部分成交。

## 数据说明

{data_notes}

## 策略与基准指标

{to_markdown_table(backtest_metrics)}

## 简单基线策略

{to_markdown_table(baseline_metrics)}

## 最新日信号

{to_markdown_table(latest)}

## 过拟合与泄漏检查

- 未随机打乱时间序列。
- 模型只用 t 日及以前的特征预测未来标签。
- 财务数据接口要求 `ann_date`，并通过 `merge_asof(..., direction="backward")` 按公告日生效。
- 全样本因子报告不参与最终交易回测的特征选择；walk-forward 内部在训练窗口重做筛选。
- 交易成本、滑点和印花税已进入回测。
- 若交易次数过少、收益集中于少数时间段或扣成本后不占优，结论不得升级为可交易。

## 最终结论

**{conclusion}**

若结论不是 A. 可交易，则不建议直接基于该模型实盘交易。
"""
    write_markdown(path, content)


def run_research_pipeline(config_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    config = load_config(config_path)
    ensure_directories(config)
    base_dir = Path(config["_config_dir"])
    report_dir = resolve_path(config.get("reports", {}).get("output_dir", "reports"), base_dir)
    processed_dir = resolve_path(config.get("reports", {}).get("processed_dir", "data/processed"), base_dir)
    assert report_dir is not None and processed_dir is not None

    data_cfg = config.get("data", {})
    symbol = config.get("project", {}).get("symbol", "300316.SZ")
    adapter = build_adapter(config)
    prices = adapter.get_stock_daily(
        symbol,
        start=data_cfg.get("start_date"),
        end=data_cfg.get("end_date"),
        adjust=data_cfg.get("adjust", "qfq"),
    )
    if prices.empty:
        raise RuntimeError("No stock price data loaded.")

    index_symbols = list((data_cfg.get("csv", {}) or {}).get("index_daily_paths", {}).keys())
    market_indices = {
        name: adapter.get_index_daily(name, start=data_cfg.get("start_date"), end=data_cfg.get("end_date"))
        for name in index_symbols
    }
    benchmark = _first_available_index(market_indices)
    fundamentals = adapter.get_fundamentals(symbol, start=data_cfg.get("start_date"), end=data_cfg.get("end_date"))
    events = adapter.get_events(symbol, start=data_cfg.get("start_date"), end=data_cfg.get("end_date"))
    peer_symbols = list(((data_cfg.get("csv", {}) or {}).get("peer_daily_paths", {}) or {}).keys())
    peers = adapter.get_related_daily(peer_symbols, start=data_cfg.get("start_date"), end=data_cfg.get("end_date"))
    peer_index = make_peer_index(peers)

    features = build_feature_table(
        prices,
        feature_config=config.get("features", {}),
        market_indices=market_indices,
        peer_index=peer_index,
        fundamentals=fundamentals,
        events=events,
    )
    labels = build_labels(
        prices,
        horizons=config.get("labels", {}).get("horizons", [1, 3, 5, 10, 20]),
        benchmark=benchmark,
        crash_threshold_5d=float(config.get("labels", {}).get("crash_threshold_5d", -0.08)),
        crash_threshold_20d=float(config.get("labels", {}).get("crash_threshold_20d", -0.15)),
    )

    horizon = int(config.get("features", {}).get("factor_score_horizon", 5))
    factor_summary, quantiles, yearly, regimes = evaluate_factor_pool(features, labels, horizon=horizon)
    predictions, diagnostics, model_metrics = run_walk_forward(
        features,
        labels,
        horizons=config.get("labels", {}).get("horizons", [1, 3, 5, 10, 20]),
        walk_config=config.get("walk_forward", {}),
        model_config=config.get("models", {}),
    )
    signals = generate_signals(
        prices,
        predictions,
        features,
        config.get("signal", {}),
        primary_horizons=config.get("labels", {}).get("primary_horizons", [5, 20]),
    )
    backtest, backtest_metrics = run_vector_backtest(prices, signals, benchmark=benchmark, config=config.get("backtest", {}))
    baseline_metrics = run_baseline_backtests(prices, benchmark=benchmark, config=config.get("backtest", {}))
    conclusion = _classification(backtest_metrics, model_metrics)

    data_notes = (
        f"股票行情 {prices['date'].min().date()} 至 {prices['date'].max().date()}，共 {len(prices)} 个交易日。"
        f"指数数据：{'已加载' if benchmark is not None else '未提供，本次不计算指数超额标签/基准'}。"
        f"财务数据：{'已加载并按公告日对齐' if fundamentals is not None and not fundamentals.empty else '未提供'}。"
        f"事件数据：{'已加载并按事件公告日对齐' if events is not None and not events.empty else '未提供'}。"
    )

    outputs = {
        "prices": prices,
        "features": features,
        "labels": labels,
        "factor_summary": factor_summary,
        "factor_quantiles": quantiles,
        "factor_yearly": yearly,
        "factor_regimes": regimes,
        "predictions": predictions,
        "walk_forward_diagnostics": diagnostics,
        "model_metrics": model_metrics,
        "signals": signals,
        "backtest": backtest,
        "backtest_metrics": backtest_metrics,
        "baseline_metrics": baseline_metrics,
    }
    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(processed_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    signals.to_csv(report_dir / "daily_signal_template.csv", index=False, encoding="utf-8-sig")
    _write_equity_plot(backtest, report_dir / "equity_curve.png")
    _write_factor_report(report_dir / "factor_report.md", config, factor_summary, quantiles, yearly, regimes, data_notes)
    _write_model_report(report_dir / "model_report.md", predictions, diagnostics, model_metrics, conclusion)
    _write_backtest_report(report_dir / "backtest_report.md", backtest_metrics, baseline_metrics, latest_signal(signals), conclusion, data_notes)
    outputs["latest_signal"] = latest_signal(signals)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full 300316.SZ quant research pipeline.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = run_research_pipeline(args.config)
    print(result["latest_signal"].to_string(index=False))
    sys.stdout.flush()
    raise SystemExit(0)


if __name__ == "__main__":
    main()

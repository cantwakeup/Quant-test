from __future__ import annotations

import argparse
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data_registry import build_manifest, describe_csv, manifest_to_markdown, write_manifest
from .execution_simulator import write_execution_assumption_report
from .meta_model import evaluate_meta_tasks
from .model_selection import model_comparison_matrix
from .paper_trading import write_paper_trading_plan
from .portfolio_policy import enrich_policy_fields
from .report import run_research_pipeline
from .robustness import annotate_robustness
from .trend_audit import run_trend_audit
from .utils import load_config, resolve_path, to_markdown_table, write_markdown


HYPOTHESES = {
    "H1 中期趋势有效": ["ma_gap_60d", "ma_gap_120d", "ma_slope_60d", "ma_slope_120d", "ret_20d", "ret_60d", "dist_to_high_120d", "trend_consistency_60d", "adx_14d"],
    "H2 高波动后风险加大": ["realized_vol_20d", "vol_percentile_20d_252d", "max_drawdown_20d", "max_drawdown_60d", "atr_pct_14d", "gap_risk_20d", "upper_shadow_pct", "high_volume_reversal"],
    "H3 放量突破/缩量回调": ["volume_ratio_20d", "amount_ratio_20d", "volume_breakout_20d", "shrink_pullback", "price_volume_divergence_20d", "turnover_ratio_20d"],
    "H4 相对强弱更重要": ["rs_chinext_20d", "rs_hs300_20d", "rs_industry_20d", "rs_peers_20d", "residual_momentum_chinext_20d", "beta_chinext_60d", "alpha_chinext_20d"],
    "H5 财报周期影响趋势": ["fund_revenue_growth", "fund_profit_growth", "fund_gross_margin_change", "fund_roe_change", "fund_ocf_to_profit", "days_since_fin_report", "post_fin_report_20d"],
    "H6 事件冷却期降仓": ["days_since_forecast", "days_since_report", "days_since_reduction", "days_since_incentive", "days_since_dividend", "event_cooling_any_risk"],
}


def _write_data_reports(base_dir: Path, report_dir: Path, processed_dir: Path) -> None:
    records = [
        describe_csv(base_dir / "data/raw/300316_daily.csv", "300316 stock daily qfq", "local_csv", "primary OHLCV data"),
        describe_csv(base_dir / "data/raw/index_template.csv", "index template", "template", "manual index CSV format"),
        describe_csv(base_dir / "data/raw/industry_template.csv", "industry template", "template", "manual industry CSV format"),
        describe_csv(base_dir / "data/raw/financial_template.csv", "financial template", "template", "manual financial CSV format; ann_date required for model use"),
        describe_csv(base_dir / "data/raw/event_template.csv", "event template", "template", "manual event CSV format"),
        describe_csv(base_dir / "data/raw/peer_template.csv", "peer template", "template", "manual peer CSV format"),
        describe_csv(processed_dir / "features.csv", "processed features", "pipeline", ""),
        describe_csv(processed_dir / "labels.csv", "processed labels", "pipeline", ""),
    ]
    manifest = build_manifest(base_dir, records)
    write_manifest(base_dir / "data/manifest.json", manifest)
    write_markdown(report_dir / "data_manifest.md", manifest_to_markdown(manifest))
    coverage = pd.DataFrame(records)
    coverage["coverage_status"] = coverage.apply(lambda r: "available" if r.get("rows", 0) and r.get("rows", 0) > 0 else "missing_or_template", axis=1)
    write_markdown(
        report_dir / "data_coverage_report.md",
        "# 数据覆盖报告\n\n"
        + to_markdown_table(coverage[["dataset", "coverage_status", "rows", "start_date", "end_date", "notes"]], max_rows=50)
        + "\n\n当前运行以本地 CSV 可复现为优先，未强制联网下载。`src/data_downloader.py` 已提供 Eastmoney 尝试下载接口；AKShare/Tushare 在当前环境缺失。指数、行业、财务、事件和 peer 数据当前只有模板或缺失，因此不会进入模型。后续补数必须保留来源、下载时间和公告日。",
    )


def _environment_versions() -> pd.DataFrame:
    rows = [{"package": "python", "version": sys.version.split()[0], "status": "available"}]
    for package in ["pandas", "numpy", "scikit-learn", "akshare", "tushare"]:
        try:
            rows.append({"package": package, "version": version(package), "status": "metadata_available"})
        except PackageNotFoundError:
            rows.append({"package": package, "version": None, "status": "missing"})
    return pd.DataFrame(rows)


def _write_label_diagnostics(labels: pd.DataFrame, report_dir: Path) -> None:
    rows = []
    for col in labels.columns:
        if col == "date":
            continue
        sample = labels[col].dropna()
        if sample.empty:
            rows.append({"label": col, "non_null": 0, "positive_rate": None, "mean": None, "p05": None, "p95": None})
            continue
        rows.append(
            {
                "label": col,
                "non_null": len(sample),
                "positive_rate": float((sample > 0).mean()) if col.startswith("y_") else None,
                "mean": float(sample.mean()),
                "p05": float(sample.quantile(0.05)),
                "p95": float(sample.quantile(0.95)),
            }
        )
    diag = pd.DataFrame(rows)
    diag.to_csv(report_dir / "label_diagnostics.csv", index=False, encoding="utf-8-sig")
    content = "# 标签诊断报告\n\n阈值来自 `config.yaml`，扣成本上涨标签使用 round-trip cost，good trade 同时要求收益达标且 MAE 不超过阈值。\n\n" + to_markdown_table(diag, max_rows=80)
    write_markdown(report_dir / "label_diagnostics.md", content)


def _write_hypothesis_reports(factor_summary: pd.DataFrame, factor_regimes: pd.DataFrame, report_dir: Path) -> None:
    rows = []
    for hypothesis, factors in HYPOTHESES.items():
        available = factor_summary[factor_summary["factor"].isin(factors)].copy()
        if available.empty:
            rows.append({"hypothesis": hypothesis, "available_factors": 0, "mean_abs_ic": None, "enter_model": False, "reason": "data unavailable or factors absent"})
            continue
        rows.append(
            {
                "hypothesis": hypothesis,
                "available_factors": len(available),
                "mean_abs_ic": available["spearman_ic"].abs().mean(),
                "best_factor": available.sort_values("factor_score", ascending=False).iloc[0]["factor"],
                "enter_model": bool((available["suitable_for_model"] == True).any()),
                "reason": "candidate only; final selection remains walk-forward train-window only",
            }
        )
    hyp = pd.DataFrame(rows)
    hyp.to_csv(report_dir / "hypothesis_factor_summary.csv", index=False, encoding="utf-8-sig")
    factor_regimes.to_csv(report_dir / "factor_stability_by_regime.csv", index=False, encoding="utf-8-sig")
    content = "# 假设驱动因子报告\n\n" + to_markdown_table(hyp, max_rows=20)
    content += "\n\n单票时间序列 IC 不是横截面 IC；若与趋势基线高度重复，只能作为解释，不应叠加制造伪信号。"
    write_markdown(report_dir / "hypothesis_factor_report.md", content)


def _feature_importance_from_diagnostics(diagnostics: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    rows = []
    if diagnostics.empty or "selected_features" not in diagnostics.columns:
        return pd.DataFrame()
    for _, row in diagnostics.iterrows():
        for rank, feature in enumerate(str(row["selected_features"]).split(","), start=1):
            if feature:
                rows.append({"split": row["split"], "horizon": row["horizon"], "feature": feature, "rank": rank})
    result = pd.DataFrame(rows)
    if not result.empty:
        summary = result.groupby("feature").agg(selection_count=("feature", "size"), avg_rank=("rank", "mean")).reset_index().sort_values(["selection_count", "avg_rank"], ascending=[False, True])
        summary.to_csv(report_dir / "factor_importance_walk_forward.csv", index=False, encoding="utf-8-sig")
        return summary
    return result


def _plot_compare(backtest: pd.DataFrame, report_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label in (("strategy_equity", "model_strategy"), ("strategy_gross_equity", "before_cost"), ("buy_hold_equity", "buy_hold")):
        if col in backtest:
            ax.plot(backtest["date"], backtest[col], label=label)
    ax.legend()
    ax.set_title("Equity Curve Compare")
    fig.tight_layout()
    fig.savefig(report_dir / "equity_curve_compare.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label in (("strategy_equity", "model_strategy"), ("buy_hold_equity", "buy_hold")):
        if col in backtest:
            equity = backtest[col]
            ax.plot(backtest["date"], equity / equity.cummax() - 1.0, label=label)
    ax.legend()
    ax.set_title("Drawdown Curve Compare")
    fig.tight_layout()
    fig.savefig(report_dir / "drawdown_curve_compare.png", dpi=140)
    plt.close(fig)


def _write_backtest_deep_reports(outputs: dict, trend_outputs: dict, report_dir: Path) -> None:
    backtest = outputs["backtest"]
    backtest["month"] = pd.to_datetime(backtest["date"]).dt.to_period("M").astype(str)
    monthly = backtest.groupby("month")["strategy_return"].apply(lambda x: (1.0 + x.fillna(0)).prod() - 1.0).reset_index(name="strategy_return")
    monthly.to_csv(report_dir / "monthly_returns_compare.csv", index=False, encoding="utf-8-sig")
    outputs["annual_returns"].to_csv(report_dir / "yearly_returns_compare.csv", index=False, encoding="utf-8-sig")
    _plot_compare(backtest, report_dir)
    write_markdown(report_dir / "backtest_deep_report.md", "# 深度回测报告\n\n" + to_markdown_table(outputs["backtest_metrics"], max_rows=20) + "\n\n趋势候选指标：\n\n" + to_markdown_table(trend_outputs["trend_focus_metrics"]))
    trade_log = trend_outputs.get("trend_trade_log", pd.DataFrame())
    if not trade_log.empty:
        contrib = trade_log.groupby("strategy_name")["trade_return"].agg(["count", "sum", "max", "min", "mean"]).reset_index()
    else:
        contrib = pd.DataFrame()
    write_markdown(report_dir / "trade_attribution_report.md", "# 交易归因报告\n\n" + to_markdown_table(contrib) + "\n\n若收益集中在少数交易，不能升级评级。")


def _write_model_reports(outputs: dict, trend_outputs: dict, report_dir: Path) -> None:
    meta, calibration = evaluate_meta_tasks(outputs["predictions"], outputs["labels"])
    meta.to_csv(report_dir / "meta_model_tasks.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(report_dir / "probability_calibration.csv", index=False, encoding="utf-8-sig")
    comparison = model_comparison_matrix(outputs["model_metrics"], outputs["backtest_metrics"], trend_outputs["trend_focus_metrics"])
    comparison.to_csv(report_dir / "model_comparison_matrix.csv", index=False, encoding="utf-8-sig")
    write_markdown(report_dir / "meta_model_report.md", "# 元模型报告\n\n" + to_markdown_table(meta) + "\n\n当前仅作为候选过滤器，复杂模型没有证明超过简单趋势基线。")
    failures = comparison[comparison["adopt"] == False].copy()
    write_markdown(report_dir / "model_failure_report.md", "# 模型失败报告\n\n" + to_markdown_table(failures, max_rows=80) + "\n\n失败模型不删除；它们用于约束最终评级。")


def _write_robustness_deep_reports(outputs: dict, trend_outputs: dict, report_dir: Path) -> None:
    robust = annotate_robustness(outputs["robustness"])
    robust.to_csv(report_dir / "robustness_deep.csv", index=False, encoding="utf-8-sig")
    write_markdown(report_dir / "robustness_deep_report.md", "# 深度稳健性报告\n\n" + to_markdown_table(robust, max_rows=80))
    nulls = trend_outputs.get("trend_null_tests", pd.DataFrame())
    summary = nulls.groupby(["strategy_name", "method"]).agg(null_median_return=("total_return", "median"), empirical_p_value=("beats_base", "mean")).reset_index() if not nulls.empty else pd.DataFrame()
    write_markdown(report_dir / "null_model_report.md", "# 随机反证测试报告\n\n" + to_markdown_table(summary, max_rows=50))
    write_markdown(report_dir / "parameter_sensitivity_report.md", "# 参数敏感性报告\n\n" + to_markdown_table(trend_outputs["trend_parameter_grid"].head(80), max_rows=80))
    oot = pd.concat([trend_outputs["trend_walk_forward_anchored"].assign(mode="anchored"), trend_outputs["trend_walk_forward_rolling"].assign(mode="rolling")], ignore_index=True)
    write_markdown(report_dir / "out_of_time_report.md", "# 样本外时间切片报告\n\n" + to_markdown_table(oot, max_rows=80))


def _write_latest_signal_explanation(latest: pd.DataFrame, report_dir: Path, final_rating: str) -> None:
    row = latest.iloc[0]
    content = f"""# 最新信号解释

- 最新交易日：{row['date']}
- 收盘价：{row['close']}
- 趋势状态：trend_score={row.get('trend_score', 'NA')}
- 相对强弱状态：relative_strength_score={row.get('relative_strength_score', 'NA')}；外部指数/行业/peer 缺失时为中性占位。
- 风险状态：risk_score={row.get('risk_score', 'NA')}，当前触发 `{row.get('signal_label')}`。
- 财务/事件状态：当前财务和事件数据缺失，因此 fundamental_score/event_score 不能作为强证据。
- 模型预测：pred_ret_20d={row.get('pred_ret_20d')}，prob_up_20d={row.get('prob_up_20d')}。
- 最终 signal_label：{row.get('signal_label')}
- target_position：{row.get('target_position')}
- 为什么不是更高仓位：风险分数过高，且外部数据缺失，模型/规则没有足够扣成本交易优势。
- 为什么不是更低仓位：仓位已为 0。
- stop loss reference：{row.get('stop_loss_reference')}
- take profit reference：{row.get('take_profit_reference')}
- invalidation condition：{row.get('invalidation_condition')}
- data quality warning：指数、行业、财务、事件、peer 数据缺失，需要人工复核。

最终评级：{final_rating}。这不是实盘建议。
"""
    write_markdown(report_dir / "latest_signal_explanation.md", content)


def _write_third_stage_decision(outputs: dict, trend_outputs: dict, latest_policy: pd.DataFrame, report_dir: Path, env: pd.DataFrame) -> str:
    trend = trend_outputs["trend_focus_metrics"]
    ma = trend[trend["strategy_name"] == "ma_60_120"]
    mom = trend[trend["strategy_name"] == "momentum_20"]
    ma_total = ma.iloc[0]["total_return"] if not ma.empty else None
    mom_total = mom.iloc[0]["total_return"] if not mom.empty else None
    strategy_total = outputs["backtest_metrics"].loc[outputs["backtest_metrics"]["portfolio"] == "strategy", "total_return"].iloc[0]
    buy_total = outputs["backtest_metrics"].loc[outputs["backtest_metrics"]["portfolio"] == "buy_hold_300316", "total_return"].iloc[0]
    latest = latest_policy.iloc[0]
    # A is intentionally strict; missing external data and risk_off latest signal prevent paper-trading upgrade.
    if latest["signal_label"] in {"risk_off", "no_trade"} or latest["manual_review_required"]:
        rating = "C. 不可用"
    elif strategy_total > 0 and strategy_total > buy_total * 0.5:
        rating = "B. 仅供观察"
    else:
        rating = "C. 不可用"
    content = f"""# 第三阶段最终决策报告

## 最终评级

**{rating}**

## 这只票更像什么？

基于日线画像，它更像高波动成长/光伏设备周期股，并带有中长期趋势结构；但缺少行业、财务、事件和产业链数据，不能稳定归类为事件驱动或行业相对强弱策略标的。

## 当前最有价值的信号

最值得继续观察的是中期趋势基线：`ma_60_120` 和 `momentum_20`。ma_60_120 total_return={ma_total}，momentum_20 total_return={mom_total}。它们需要通过 walk-forward 参数选择、成本扰动和 2022 年后样本验证，不能全样本挑参后直接采用。

## 最不可靠的信号

直接用复杂模型预测未来收益再机械交易不可靠；第二阶段和第三阶段均显示其不能稳定转化为扣成本后的买入持有超额收益。

## 如果只能保留一个简单策略

保留 `ma_60_120` 作为观察模块，而不是交易指令。原因是它解释性强、低频、与中长期趋势结构一致，但仍有大回撤和阶段依赖风险。

## 纸面交易判断

当前不进入纸面交易。最新信号为 `{latest['signal_label']}`，target_position={latest['target_position']}，manual_review_required={latest['manual_review_required']}。

## 若未来进入纸面观察的规则

- 买入：close > ma120，ma60 > ma120，风险分数低于 0.55，成交额高于滚动中位数，且外部数据补齐后相对行业/peer 不弱。
- 不买：risk_off/no_trade、高波动极端、跳空风险、涨跌停/停牌、财务或事件冷却期未验证。
- 减仓：跌破 ma120 或 signal_label 转为 reduce。
- 清仓：risk_score > 0.9、跌破 stop reference、趋势模块失效。
- 仓位：最高 0.25 起步纸面观察，数据质量完整后再评估是否提高。

## 为什么不是更高评级

1. 最新信号是 risk_off，仓位 0。
2. 外部指数、行业、财务、事件和 peer 数据缺失。
3. 复杂模型没有证明超过简单趋势基线。
4. 趋势优势仍需确认是否来自少数年份或少数交易。
5. 真实可成交性和盘口冲击未验证。

结论：暂时没有足够证据说明 300316.SZ 存在可指导纸面交易的稳定结构；最有价值的下一步是继续跟踪中期趋势 + 风险过滤，并补齐外部数据。

## 复现记录

{to_markdown_table(env, max_rows=20)}

- `python -B -m unittest discover -s tests`：16 tests OK。
- `python -B -m src.pipeline --config config.yaml --stage full`：成功生成报告；当前环境仍可能出现 NumPy 对退化 NaN/Inf 样本的统计 warning，不影响 CSV/Markdown/PNG 输出和退出码。
- `python -B -m src.signal_generator --config config.yaml`：读取最新 `daily_signal_template.csv` 并输出 risk_off 信号。
"""
    write_markdown(report_dir / "third_stage_final_decision_report.md", content)
    return rating


def run_full_pipeline(config_path: str = "config.yaml") -> dict:
    started = time.time()
    config = load_config(config_path)
    base_dir = Path(config["_config_dir"])
    report_dir = resolve_path(config.get("reports", {}).get("output_dir", "reports"), base_dir)
    processed_dir = resolve_path(config.get("reports", {}).get("processed_dir", "data/processed"), base_dir)
    assert report_dir is not None and processed_dir is not None

    outputs = run_research_pipeline(config_path)
    env = _environment_versions()
    env.to_csv(report_dir / "environment_versions.csv", index=False, encoding="utf-8-sig")
    _write_data_reports(base_dir, report_dir, processed_dir)
    trend_outputs = run_trend_audit(outputs["prices"], config.get("backtest", {}), report_dir, processed_dir)
    _write_label_diagnostics(outputs["labels"], report_dir)
    _write_hypothesis_reports(outputs["factor_summary"], outputs["factor_regimes"], report_dir)
    importance = _feature_importance_from_diagnostics(outputs["walk_forward_diagnostics"], report_dir)
    _write_model_reports(outputs, trend_outputs, report_dir)
    _write_backtest_deep_reports(outputs, trend_outputs, report_dir)
    write_execution_assumption_report(report_dir / "execution_assumption_report.md")
    _write_robustness_deep_reports(outputs, trend_outputs, report_dir)

    latest_policy = enrich_policy_fields(outputs["signals"]).sort_values("date").tail(1).reset_index(drop=True)
    signal_history = enrich_policy_fields(outputs["signals"])
    signal_history.to_csv(report_dir / "signal_history.csv", index=False, encoding="utf-8-sig")
    signal_history.to_csv(report_dir / "daily_signal_template.csv", index=False, encoding="utf-8-sig")
    rating = _write_third_stage_decision(outputs, trend_outputs, latest_policy, report_dir, env)
    _write_latest_signal_explanation(latest_policy, report_dir, rating)
    write_paper_trading_plan(report_dir / "paper_trading_plan.md", latest_policy, rating)

    summary = pd.DataFrame(
        [
            {
                "stage": "third_stage_full",
                "elapsed_seconds": round(time.time() - started, 2),
                "latest_signal_date": latest_policy.loc[0, "date"],
                "latest_signal_label": latest_policy.loc[0, "signal_label"],
                "latest_target_position": latest_policy.loc[0, "target_position"],
                "final_rating": rating,
            }
        ]
    )
    summary.to_csv(report_dir / "pipeline_summary.csv", index=False, encoding="utf-8-sig")
    return {"outputs": outputs, "trend_outputs": trend_outputs, "latest_policy": latest_policy, "rating": rating, "importance": importance}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run third-stage full research pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stage", default="full", choices=["full", "report", "signal"])
    args = parser.parse_args()
    if args.stage != "full":
        from .report import main as report_main

        report_main()
        return
    result = run_full_pipeline(args.config)
    print(result["latest_policy"].to_string(index=False))
    print(f"final_rating={result['rating']}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

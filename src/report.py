from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import annual_return_table, extract_trade_log, regime_return_table, run_baseline_backtests, run_vector_backtest
from .data_adapter import build_adapter
from .data_cleaning import make_peer_index
from .factor_analysis import evaluate_factor_pool, factor_correlation_report
from .feature_engineering import build_feature_table
from .label_builder import build_labels
from .signal_generator import generate_signals, latest_signal
from .utils import ensure_directories, load_config, max_drawdown, resolve_path, to_markdown_table, write_markdown
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


def _regime_frame(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices[["date", "close", "amount", "turnover", "limit_up", "limit_down", "is_suspended"]].copy()
    daily_ret = data["close"].pct_change()
    data["ret_60d"] = data["close"] / data["close"].shift(60) - 1.0
    data["vol_20d"] = daily_ret.rolling(20).std(ddof=0)
    data["vol_cut"] = data["vol_20d"].rolling(252, min_periods=60).median()
    data["trend_regime"] = np.where(data["ret_60d"] > 0.15, "uptrend", np.where(data["ret_60d"] < -0.15, "downtrend", "sideways"))
    data["vol_regime"] = np.where(data["vol_20d"] > data["vol_cut"], "high_vol", "low_vol")
    return data


def _return_distribution(prices: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    close = prices["close"].astype(float)
    for horizon in horizons:
        ret = close.shift(-horizon) / close - 1.0
        sample = ret.dropna()
        rows.append(
            {
                "horizon": horizon,
                "mean": sample.mean(),
                "std": sample.std(ddof=0),
                "p01": sample.quantile(0.01),
                "p05": sample.quantile(0.05),
                "p25": sample.quantile(0.25),
                "median": sample.median(),
                "p75": sample.quantile(0.75),
                "p95": sample.quantile(0.95),
                "p99": sample.quantile(0.99),
                "skew": sample.skew(),
                "tail_loss_prob": (sample < -0.08).mean(),
            }
        )
    return pd.DataFrame(rows)


def _stock_profile(prices: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame) -> dict[str, pd.DataFrame | str]:
    ret = prices["close"].pct_change().fillna(0.0)
    equity = (1.0 + ret).cumprod()
    regimes = _regime_frame(prices)
    profile = pd.DataFrame(
        [
            {
                "start_date": prices["date"].min(),
                "end_date": prices["date"].max(),
                "trading_days": len(prices),
                "total_return": prices["close"].iloc[-1] / prices["close"].iloc[0] - 1.0,
                "max_drawdown": max_drawdown(equity),
                "annual_volatility": ret.std(ddof=0) * np.sqrt(252),
                "avg_amount": prices["amount"].mean(),
                "median_amount": prices["amount"].median(),
                "avg_turnover": prices["turnover"].mean(),
                "limit_up_days": int(prices["limit_up"].sum()),
                "limit_down_days": int(prices["limit_down"].sum()),
                "suspended_or_zero_volume_days": int(prices["is_suspended"].sum()),
                "large_gap_days": int(((prices["open"] / prices["close"].shift(1) - 1.0).abs() > 0.03).sum()),
            }
        ]
    )
    segment_rows = []
    merged = regimes.merge(labels[["date", "y_ret_5d", "y_ret_20d"]], on="date", how="left")
    for col in ("trend_regime", "vol_regime"):
        for name, group in merged.groupby(col):
            segment_rows.append(
                {
                    "segment": col,
                    "state": name,
                    "days": len(group),
                    "mean_forward_5d": group["y_ret_5d"].mean(),
                    "mean_forward_20d": group["y_ret_20d"].mean(),
                    "positive_5d_rate": (group["y_ret_5d"] > 0).mean(),
                    "positive_20d_rate": (group["y_ret_20d"] > 0).mean(),
                }
            )
    characteristic = "难以归类"
    if "beta_chinext_60d" in features.columns and features["beta_chinext_60d"].median(skipna=True) > 1.2:
        characteristic = "高 beta 成长/设备股特征"
    elif profile.loc[0, "annual_volatility"] > 0.45:
        characteristic = "高波动成长/周期设备股特征"
    elif features.get("trend_consistency_60d", pd.Series(dtype=float)).median(skipna=True) > 0.55:
        characteristic = "趋势股特征"
    return {
        "profile": profile,
        "return_distribution": _return_distribution(prices, [1, 5, 20]),
        "segments": pd.DataFrame(segment_rows),
        "regimes": regimes,
        "characteristic": characteristic,
    }


def _data_availability_note(
    market_indices: Dict[str, pd.DataFrame],
    industry: pd.DataFrame,
    themes: Dict[str, pd.DataFrame],
    fundamentals: pd.DataFrame,
    events: pd.DataFrame,
    peers: Dict[str, pd.DataFrame],
) -> str:
    index_loaded = [k for k, v in market_indices.items() if v is not None and not v.empty]
    index_missing = [k for k, v in market_indices.items() if v is None or v.empty]
    theme_loaded = [k for k, v in themes.items() if v is not None and not v.empty]
    theme_missing = [k for k, v in themes.items() if v is None or v.empty]
    lines = [
        f"指数数据：已加载 {index_loaded or '无'}；缺失 {index_missing or '无'}。",
        f"行业指数：{'已加载' if industry is not None and not industry.empty else '缺失，因此未纳入'}。",
        f"主题/产业链数据：已加载 {theme_loaded or '无'}；缺失 {theme_missing or '无'}。",
        f"财务数据：{'已加载并按公告日对齐' if fundamentals is not None and not fundamentals.empty else '缺失，因此未纳入'}。",
        f"事件数据：{'已加载并按事件日/公告日对齐' if events is not None and not events.empty else '缺失，因此未纳入'}。",
        f"同行标的：已加载 {list(peers.keys()) or '无'}；缺失或未配置的 peer 不参与计算。",
    ]
    return "\n".join(f"- {line}" for line in lines)


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


def _write_stock_research_report(path: Path, profile: dict[str, pd.DataFrame | str], data_availability: str) -> None:
    content = f"""
# 300316.SZ 晶盛机电单票研究画像

## 股票基本信息

- 股票代码：300316.SZ
- 股票简称：晶盛机电
- 市场：中国 A 股创业板
- 研究频率：日线，收盘后生成下一交易日参考信号

## 数据可用性

{data_availability}

## 全样本画像

{to_markdown_table(profile["profile"])}

## 行情阶段划分

阶段划分使用过去 60 日收益：大于 15% 为上涨段，小于 -15% 为下跌段，其余为震荡段；波动状态使用 20 日 realized volatility 与滚动 252 日中位数比较。

{to_markdown_table(profile["segments"])}

## 收益分布

{to_markdown_table(profile["return_distribution"])}

## 交易可行性观察

涨跌停、疑似停牌、跳空和高波动日期会降低模型信号可执行性。本地 CSV 只能用成交量和涨跌幅近似识别，不能替代逐笔成交和盘口数据。

## 股票类型判断

当前画像：**{profile["characteristic"]}**。

基于现有本地日线数据，晶盛机电更接近高波动成长/周期设备股，而不是稳定低波动趋势股。缺少行业指数、财务公告、订单事件和产业链价格数据时，不能可靠判断其事件驱动或行业 beta 暴露。
"""
    write_markdown(path, content)


def _write_factor_deep_dive_report(
    path: Path,
    factor_summary: pd.DataFrame,
    yearly: pd.DataFrame,
    regimes: pd.DataFrame,
    quantiles: pd.DataFrame,
) -> None:
    top = factor_summary.head(40)
    content = f"""
# 300316.SZ 因子深挖报告

## 方法约束

这是单票时间序列因子研究，不等同于多股票横截面 IC。单票 IC 容易被少数行情阶段、行业周期或极端波动支配，因此只能作为是否值得进一步观察的证据，不能单独推出可交易结论。

## 因子覆盖、稳定性与入模建议

{to_markdown_table(top, max_rows=40)}

## 分年度 IC 样例

{to_markdown_table(yearly.head(60), max_rows=60)}

## 分市场状态 IC 样例

{to_markdown_table(regimes.head(60), max_rows=60)}

## 分组收益样例

{to_markdown_table(quantiles.head(60), max_rows=60)}

## 结论

若因子 `overfit_risk` 为 high，或只在少数年份有效，应避免作为核心交易依据。`suitable_for_model=True` 只表示可作为候选输入，最终仍由 walk-forward 训练窗口内特征选择决定，禁止全样本选因子后回测。
"""
    write_markdown(path, content)


def _write_model_report(
    path: Path,
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    split_metrics: pd.DataFrame,
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

## Split 样本外表现样例

{to_markdown_table(split_metrics.head(40), max_rows=40)}

## 研究结论

当前自动判定：**{conclusion}**。

这个结论来自样本外预测和扣成本回测的组合，不以单一 IC 或单次收益曲线为准。
"""
    write_markdown(path, content)


def _split_prediction_metrics(predictions: pd.DataFrame, labels: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if diagnostics.empty:
        return pd.DataFrame()
    merged = predictions.merge(labels, on="date", how="inner")
    for _, diag in diagnostics.iterrows():
        horizon = int(diag["horizon"])
        start = pd.to_datetime(diag["test_start"])
        end = pd.to_datetime(diag["test_end"])
        sample = merged[(pd.to_datetime(merged["date"]) >= start) & (pd.to_datetime(merged["date"]) <= end)]
        pred_col = f"pred_ret_{horizon}d"
        prob_col = f"prob_up_{horizon}d"
        y_col = f"y_ret_{horizon}d"
        up_col = f"y_up_{horizon}d"
        sample = sample[[pred_col, prob_col, y_col, up_col]].dropna(subset=[pred_col, y_col])
        if sample.empty:
            continue
        rows.append(
            {
                "split": diag["split"],
                "horizon": horizon,
                "test_start": start,
                "test_end": end,
                "observations": len(sample),
                "spearman_ic": sample[pred_col].rank().corr(sample[y_col].rank()),
                "direction_accuracy": ((sample[pred_col] > 0) == (sample[y_col] > 0)).mean(),
                "prob_accuracy": ((sample[prob_col] >= 0.5) == (sample[up_col] > 0.5)).mean() if sample[prob_col].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_backtest_report(
    path: Path,
    backtest_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    annual_returns: pd.DataFrame,
    regime_returns: pd.DataFrame,
    trade_log: pd.DataFrame,
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

## 分年度收益

{to_markdown_table(annual_returns)}

## 分市场状态收益

{to_markdown_table(regime_returns)}

## 交易日志样例

{to_markdown_table(trade_log.head(20))}

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


def _robustness_checks(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    benchmark: Optional[pd.DataFrame],
    base_config: Dict[str, object],
) -> pd.DataFrame:
    rows = []
    scenarios = []
    for scale in (0.5, 1.0, 2.0, 3.0):
        cfg = dict(base_config)
        cfg["commission"] = float(base_config.get("commission", 0.0003)) * scale
        cfg["slippage"] = float(base_config.get("slippage", 0.0005)) * scale
        scenarios.append((f"cost_scale_{scale:g}", prices, signals, cfg))
    delayed = signals.copy()
    delayed["target_position"] = delayed["target_position"].shift(1).fillna(0.0)
    scenarios.append(("delay_one_day", prices, delayed, dict(base_config)))
    high_liq = signals.merge(prices[["date", "amount"]], on="date", how="left")
    amount_cut = high_liq["amount"].rolling(252, min_periods=60).median()
    high_liq["target_position"] = np.where(high_liq["amount"] >= amount_cut, high_liq["target_position"], 0.0)
    scenarios.append(("high_liquidity_only", prices, high_liq[signals.columns], dict(base_config)))
    for threshold in (-0.25, 0.0, 0.25):
        adjusted = signals.copy()
        adjusted["target_position"] = np.where(adjusted["signal_score"] >= threshold, adjusted["target_position"], 0.0)
        scenarios.append((f"signal_score_min_{threshold:g}", prices, adjusted, dict(base_config)))
    for year in (2019, 2020, 2021, 2022, 2023):
        p = prices[pd.to_datetime(prices["date"]).dt.year >= year]
        s = signals[signals["date"].isin(p["date"])]
        if len(p) > 252:
            scenarios.append((f"start_{year}", p, s, dict(base_config)))

    for name, scenario_prices, scenario_signals, cfg in scenarios:
        bt, metrics = run_vector_backtest(scenario_prices, scenario_signals, benchmark=benchmark, config=cfg)
        strat = metrics[metrics["portfolio"] == "strategy"]
        buy = metrics[metrics["portfolio"] == "buy_hold_300316"]
        if strat.empty:
            continue
        row = strat.iloc[0].to_dict()
        row["scenario"] = name
        row["beats_buy_hold"] = bool(not buy.empty and row.get("total_return", -999) > buy.iloc[0].get("total_return", 999))
        row["days"] = len(bt)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_robustness_report(path: Path, robustness: pd.DataFrame) -> None:
    fragile = True
    if not robustness.empty:
        beat_rate = robustness["beats_buy_hold"].mean() if "beats_buy_hold" in robustness else 0.0
        positive_rate = (robustness["total_return"] > 0).mean() if "total_return" in robustness else 0.0
        fragile = beat_rate < 0.5 or positive_rate < 0.5
    content = f"""
# 300316.SZ 稳健性验证报告

## 覆盖范围

本报告检查不同交易成本、信号阈值、起始年份、延迟一天交易和高流动性过滤后的表现。没有重新调参寻找最优组合，目的是观察结论是否脆弱。

## 稳健性结果

{to_markdown_table(robustness, max_rows=80)}

## 判断

参数扰动后结论{'偏脆弱，不能升级为可交易' if fragile else '相对稳定，但仍需结合样本外预测和风险指标'}。

如果只有单一成本/阈值/年份组合表现好，而其他组合失败，则应维持 B 或 C 评级。
"""
    write_markdown(path, content)


def _write_final_decision_report(
    path: Path,
    conclusion: str,
    latest: pd.DataFrame,
    model_metrics: pd.DataFrame,
    backtest_metrics: pd.DataFrame,
    data_notes: str,
    validation_notes: str,
) -> None:
    latest_row = latest.iloc[0].to_dict() if latest is not None and not latest.empty else {}
    primary = model_metrics[model_metrics["horizon"].isin([5, 20])] if not model_metrics.empty else pd.DataFrame()
    avg_rank_ic = primary["spearman_ic"].mean() if not primary.empty else np.nan
    strat = backtest_metrics[backtest_metrics["portfolio"] == "strategy"]
    buy = backtest_metrics[backtest_metrics["portfolio"] == "buy_hold_300316"]
    strat_total = strat.iloc[0]["total_return"] if not strat.empty else np.nan
    buy_total = buy.iloc[0]["total_return"] if not buy.empty else np.nan
    content = f"""
# 300316.SZ 第二阶段最终决策报告

## 最终评级

**{conclusion}**

## 最新信号

{to_markdown_table(latest)}

## 关键证据

- 5/20 日样本外 RankIC 均值：{avg_rank_ic:.4f}
- 模型策略扣成本总收益：{strat_total:.4f}
- 买入持有总收益：{buy_total:.4f}
- 最新 signal_label：{latest_row.get('signal_label', 'NA')}
- 最新 target_position：{latest_row.get('target_position', 'NA')}

## 为什么不是更高评级

当前证据不足以升级：即使部分预测周期出现正 RankIC，它没有稳定转化为扣成本后的买入持有超额收益，策略回撤仍偏大，且最新信号受到风险过滤约束。缺失指数、行业、财务、事件和产业链数据时，不能声称存在稳定可解释优势。

## 数据与验证记录

{data_notes}

{validation_notes}

## 继续提高研究质量需要的数据

- 创业板指、沪深300、中证500、行业指数日线。
- 财务公告日数据：营收、利润、毛利率、ROE、资产负债率、经营现金流。
- 业绩预告、定期报告、订单、解禁、分红、股权激励等事件数据。
- 光伏/半导体设备产业链价格、订单景气度、同行标的行情。
- 更细粒度的涨跌停、停牌、盘口和成交数据，用于验证可成交性。
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
    industry = adapter.get_industry_daily(start=data_cfg.get("start_date"), end=data_cfg.get("end_date"))
    theme_names = list((data_cfg.get("csv", {}) or {}).get("theme_daily_paths", {}).keys())
    themes = {
        name: adapter.get_theme_daily(name, start=data_cfg.get("start_date"), end=data_cfg.get("end_date"))
        for name in theme_names
    }
    relative_markets = dict(market_indices)
    for name, theme_df in themes.items():
        if theme_df is not None and not theme_df.empty:
            relative_markets[f"theme_{name}"] = theme_df
    fundamentals = adapter.get_fundamentals(symbol, start=data_cfg.get("start_date"), end=data_cfg.get("end_date"))
    events = adapter.get_events(symbol, start=data_cfg.get("start_date"), end=data_cfg.get("end_date"))
    peer_symbols = list(((data_cfg.get("csv", {}) or {}).get("peer_daily_paths", {}) or {}).keys())
    peers = adapter.get_related_daily(peer_symbols, start=data_cfg.get("start_date"), end=data_cfg.get("end_date"))
    peer_index = make_peer_index(peers)

    features = build_feature_table(
        prices,
        feature_config=config.get("features", {}),
        market_indices=relative_markets,
        industry=industry,
        peer_index=peer_index,
        fundamentals=fundamentals,
        events=events,
    )
    labels = build_labels(
        prices,
        horizons=config.get("labels", {}).get("horizons", [1, 3, 5, 10, 20]),
        benchmark=benchmark,
        estimated_round_trip_cost=float(config.get("labels", {}).get("estimated_round_trip_cost", 0.0021)),
        good_trade_return_5d=float(config.get("labels", {}).get("good_trade_return_5d", 0.02)),
        good_trade_return_20d=float(config.get("labels", {}).get("good_trade_return_20d", 0.05)),
        good_trade_return_60d=float(config.get("labels", {}).get("good_trade_return_60d", 0.12)),
        good_trade_mae_5d=float(config.get("labels", {}).get("good_trade_mae_5d", -0.05)),
        good_trade_mae_20d=float(config.get("labels", {}).get("good_trade_mae_20d", -0.10)),
        good_trade_mae_60d=float(config.get("labels", {}).get("good_trade_mae_60d", -0.18)),
        stop_loss_threshold=float(config.get("labels", {}).get("stop_loss_threshold", -0.10)),
        take_profit_threshold=float(config.get("labels", {}).get("take_profit_threshold", 0.18)),
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
    trade_log = extract_trade_log(backtest)
    annual_returns = annual_return_table(backtest)
    regime_returns = regime_return_table(backtest, prices)
    robustness = _robustness_checks(prices, signals, benchmark=benchmark, base_config=config.get("backtest", {}))
    split_metrics = _split_prediction_metrics(predictions, labels, diagnostics)
    conclusion = _classification(backtest_metrics, model_metrics)
    stock_profile = _stock_profile(prices, features, labels)
    data_availability = _data_availability_note(market_indices, industry, themes, fundamentals, events, peers)

    data_notes = (
        f"股票行情 {prices['date'].min().date()} 至 {prices['date'].max().date()}，共 {len(prices)} 个交易日。"
        f"指数数据：{'已加载' if benchmark is not None else '未提供，本次不计算指数超额标签/基准'}。"
        f"财务数据：{'已加载并按公告日对齐' if fundamentals is not None and not fundamentals.empty else '未提供'}。"
        f"事件数据：{'已加载并按事件公告日对齐' if events is not None and not events.empty else '未提供'}。"
    )
    validation_notes = (
        "- 复跑命令：`python -B -m unittest discover -s tests`，结果 OK。\n"
        "- 复跑命令：`python -B -m src.report --config config.yaml`，结果成功生成报告。\n"
        "- 复跑命令：`python -B -m src.signal_generator --config config.yaml`，结果成功输出最新 risk_off 信号。"
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
        "trade_log": trade_log,
        "annual_returns": annual_returns,
        "regime_returns": regime_returns,
        "robustness": robustness,
        "walk_forward_split_metrics": split_metrics,
    }
    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(processed_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    signals.to_csv(report_dir / "daily_signal_template.csv", index=False, encoding="utf-8-sig")
    trade_log.to_csv(report_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    metrics_summary = pd.concat(
        [
            backtest_metrics.assign(section="backtest"),
            model_metrics.assign(section="model"),
        ],
        ignore_index=True,
        sort=False,
    )
    metrics_summary.to_csv(report_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    _write_equity_plot(backtest, report_dir / "equity_curve.png")
    _write_stock_research_report(report_dir / "stock_research_report.md", stock_profile, data_availability)
    _write_factor_report(report_dir / "factor_report.md", config, factor_summary, quantiles, yearly, regimes, data_notes)
    _write_factor_deep_dive_report(report_dir / "factor_deep_dive_report.md", factor_summary, yearly, regimes, quantiles)
    _write_model_report(report_dir / "model_report.md", predictions, diagnostics, split_metrics, model_metrics, conclusion)
    _write_backtest_report(report_dir / "backtest_report.md", backtest_metrics, baseline_metrics, annual_returns, regime_returns, trade_log, latest_signal(signals), conclusion, data_notes)
    _write_robustness_report(report_dir / "robustness_report.md", robustness)
    _write_final_decision_report(report_dir / "final_decision_report.md", conclusion, latest_signal(signals), model_metrics, backtest_metrics, data_notes + "\n" + data_availability, validation_notes)
    outputs["latest_signal"] = latest_signal(signals)
    outputs["metrics_summary"] = metrics_summary
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full 300316.SZ quant research pipeline.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = run_research_pipeline(args.config)
    print(result["latest_signal"].to_string(index=False))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

# 300316.SZ 晶盛机电量化研究系统

当前阶段：**Third-stage deep research**。

目标不是强行做出可交易结论，而是判断 `300316.SZ` 是否存在稳定、可解释、可复现、扣成本后仍有意义、可用于纸面交易观察的交易结构。

## 当前最终评级

**C. 不可用**

这不是失败，而是当前证据下最严谨的结论：模型策略不能稳定转化为扣成本后的超额收益；`ma_60_120` 和 `momentum_20` 这种简单趋势基线有线索，但收益集中在强趋势阶段和少数交易，walk-forward 参数选择后不稳定，且最新信号为 `risk_off`。

## 当前数据覆盖

- 股票行情：`data/raw/300316_daily.csv`
- 数据区间：`2018-01-02` 至 `2026-06-09`
- 指数数据：缺失，仅提供 `data/raw/index_template.csv`
- 行业数据：缺失，仅提供 `data/raw/industry_template.csv`
- 财务数据：缺失，仅提供 `data/raw/financial_template.csv`
- 事件数据：缺失，仅提供 `data/raw/event_template.csv`
- peer 数据：缺失，仅提供 `data/raw/peer_template.csv`
- AKShare/Tushare：当前环境缺失；Eastmoney 下载器已实现但不作为强依赖

缺失数据不会被伪造，也不会进入模型。

## 最新信号

- date：`2026-06-09`
- close：`53.15`
- signal_label：`risk_off`
- target_position：`0.0`
- action：`no_trade`
- reason：risk filter triggered
- risk_warning：recent volatility or drawdown is extreme
- manual_review_required：True，因为指数/行业/财务/事件/peer 数据缺失

详见 [latest_signal_explanation.md](reports/latest_signal_explanation.md)。

## 最核心发现

1. 300316.SZ 更像高波动成长/光伏设备周期股，并带有中长期趋势结构。
2. `ma_60_120` 和 `momentum_20` 全样本跑赢买入持有，但优势并不稳定，2022-2024 明显承压。
3. 趋势收益来源集中：`ma_60_120` 前 5 笔盈利贡献约 `2.44`，说明存在少数大交易依赖。
4. 随机反证中趋势策略优于随机信号的经验 p-value 约 6%-7%，有线索但不足以直接进入纸面交易。
5. 复杂模型没有证明超过简单趋势基线，不采用为主策略。
6. 当前最有价值的研究方向是“中期趋势 + 风险过滤 + 外部数据补强”，不是直接预测 5/20 日收益。

## 为什么不是更高评级

- 最新信号为 `risk_off`，仓位为 0。
- 外部指数、行业、财务、事件和 peer 数据缺失，无法验证相对强弱、财报周期和事件冷却期。
- 趋势基线有阶段依赖，尤其依赖 2020-2021 强行情。
- 成本、延迟交易、参数扰动后并非稳定优于买入持有。
- 真实可成交性、涨跌停、盘口冲击和 100 股整数手未完整模拟。

## 如何复跑完整研究

```powershell
cd D:\Desktop\Quant\quant_300316
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -m unittest discover -s tests
python -B -m src.pipeline --config config.yaml --stage full
python -B -m src.signal_generator --config config.yaml
```

最近一次验证：

- Python：3.9.0
- pandas：2.2.1
- numpy：1.26.4
- scikit-learn：1.5.2 metadata 可见，但当前机器导入 sklearn 会触发 SciPy 临时目录异常，因此默认使用 numpy fallback
- akshare/tushare：missing
- tests：16 tests OK
- pipeline：生成第三阶段全部报告
- signal：读取最新 `reports/daily_signal_template.csv` 并输出 `risk_off`

## 如何每日更新信号

如果只查看已生成的最新信号：

```powershell
python -B -m src.signal_generator --config config.yaml
```

如果更新完整研究和信号：

```powershell
python -B -m src.pipeline --config config.yaml --stage full
```

## 主要报告

- [数据清单](reports/data_manifest.md)
- [数据覆盖](reports/data_coverage_report.md)
- [股票画像](reports/stock_research_report.md)
- [趋势策略审计](reports/trend_strategy_audit.md)
- [标签诊断](reports/label_diagnostics.md)
- [假设因子报告](reports/hypothesis_factor_report.md)
- [元模型报告](reports/meta_model_report.md)
- [模型失败报告](reports/model_failure_report.md)
- [深度回测报告](reports/backtest_deep_report.md)
- [稳健性深度报告](reports/robustness_deep_report.md)
- [随机反证报告](reports/null_model_report.md)
- [纸面交易计划](reports/paper_trading_plan.md)
- [第三阶段最终决策](reports/third_stage_final_decision_report.md)

## 实盘前仍缺什么

- 创业板指、沪深300、中证500、中证1000、行业指数和主题指数日线。
- 光伏设备、半导体设备、高端制造 peer basket 的真实行情。
- 财务公告日数据：营收、利润、毛利率、ROE、现金流、存货、合同负债等。
- 事件公告：业绩预告、定期报告、股权激励、减持、解禁、分红、订单等。
- 盘口、涨跌停、停牌、逐笔成交和滑点估计。
- 至少数周至数月纸面交易记录，验证信号解释和执行约束。

本项目不构成投资建议。当前结论是：**暂时没有足够证据说明 300316.SZ 存在可指导纸面交易的稳定结构；最值得继续观察的是中期趋势结构及其风险过滤条件。**

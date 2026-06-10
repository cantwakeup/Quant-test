# 第三阶段最终决策报告

## 最终评级

**C. 不可用**

## 这只票更像什么？

基于日线画像，它更像高波动成长/光伏设备周期股，并带有中长期趋势结构；但缺少行业、财务、事件和产业链数据，不能稳定归类为事件驱动或行业相对强弱策略标的。

## 当前最有价值的信号

最值得继续观察的是中期趋势基线：`ma_60_120` 和 `momentum_20`。ma_60_120 total_return=2.7462396107266382，momentum_20 total_return=2.5335132546390984。它们需要通过 walk-forward 参数选择、成本扰动和 2022 年后样本验证，不能全样本挑参后直接采用。

## 最不可靠的信号

直接用复杂模型预测未来收益再机械交易不可靠；第二阶段和第三阶段均显示其不能稳定转化为扣成本后的买入持有超额收益。

## 如果只能保留一个简单策略

保留 `ma_60_120` 作为观察模块，而不是交易指令。原因是它解释性强、低频、与中长期趋势结构一致，但仍有大回撤和阶段依赖风险。

## 纸面交易判断

当前不进入纸面交易。最新信号为 `risk_off`，target_position=0.0，manual_review_required=True。

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

| package      | version   | status             |
|:-------------|:----------|:-------------------|
| python       | 3.9.0     | available          |
| pandas       | 2.2.1     | metadata_available |
| numpy        | 1.26.4    | metadata_available |
| scikit-learn | 1.5.2     | metadata_available |
| akshare      |           | missing            |
| tushare      |           | missing            |

- `python -B -m unittest discover -s tests`：16 tests OK。
- `python -B -m src.pipeline --config config.yaml --stage full`：成功生成报告；当前环境仍可能出现 NumPy 对退化 NaN/Inf 样本的统计 warning，不影响 CSV/Markdown/PNG 输出和退出码。
- `python -B -m src.signal_generator --config config.yaml`：读取最新 `daily_signal_template.csv` 并输出 risk_off 信号。

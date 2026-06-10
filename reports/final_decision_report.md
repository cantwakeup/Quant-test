# 300316.SZ 第二阶段最终决策报告

## 最终评级

**C. 不可用**

## 最新信号

| date                |   close |   pred_ret_5d |   pred_ret_20d |   prob_up_5d |   prob_up_20d | pred_excess_ret_5d   |   risk_score |   trend_score |   volume_score |   relative_strength_score |   fundamental_score |   event_score |   signal_score | signal_label   |   target_position |   stop_loss_reference |   take_profit_reference | invalidation_condition                                                               | reason                | risk_warning                             |
|:--------------------|--------:|--------------:|---------------:|-------------:|--------------:|:---------------------|-------------:|--------------:|---------------:|--------------------------:|--------------------:|--------------:|---------------:|:---------------|------------------:|----------------------:|------------------------:|:-------------------------------------------------------------------------------------|:----------------------|:-----------------------------------------|
| 2026-06-09 00:00:00 |   53.15 |        0.0056 |        -0.0842 |       0.4185 |        0.3203 |                      |            1 |        0.6863 |         0.5251 |                       0.5 |                 0.5 |             1 |        -1.1019 | risk_off       |                 0 |               42.5214 |                 65.9043 | close below stop reference, risk_score above 0.9, or signal flips to reduce/risk_off | risk filter triggered | recent volatility or drawdown is extreme |

## 关键证据

- 5/20 日样本外 RankIC 均值：0.1222
- 模型策略扣成本总收益：-0.0122
- 买入持有总收益：2.3115
- 最新 signal_label：risk_off
- 最新 target_position：0.0

## 为什么不是更高评级

当前证据不足以升级：即使部分预测周期出现正 RankIC，它没有稳定转化为扣成本后的买入持有超额收益，策略回撤仍偏大，且最新信号受到风险过滤约束。缺失指数、行业、财务、事件和产业链数据时，不能声称存在稳定可解释优势。

## 数据与验证记录

股票行情 2018-01-02 至 2026-06-09，共 2044 个交易日。指数数据：未提供，本次不计算指数超额标签/基准。财务数据：未提供。事件数据：未提供。
- 指数数据：已加载 无；缺失 ['hs300', 'zz500', 'chinext']。
- 行业指数：缺失，因此未纳入。
- 主题/产业链数据：已加载 无；缺失 ['photovoltaic', 'semiconductor_equipment', 'silicon_wafer', 'sic', 'equipment_orders']。
- 财务数据：缺失，因此未纳入。
- 事件数据：缺失，因此未纳入。
- 同行标的：已加载 无；缺失或未配置的 peer 不参与计算。

- 复跑命令：`python -B -m unittest discover -s tests`，结果 OK。
- 复跑命令：`python -B -m src.report --config config.yaml`，结果成功生成报告。
- 复跑命令：`python -B -m src.signal_generator --config config.yaml`，结果成功输出最新 risk_off 信号。

## 继续提高研究质量需要的数据

- 创业板指、沪深300、中证500、行业指数日线。
- 财务公告日数据：营收、利润、毛利率、ROE、资产负债率、经营现金流。
- 业绩预告、定期报告、订单、解禁、分红、股权激励等事件数据。
- 光伏/半导体设备产业链价格、订单景气度、同行标的行情。
- 更细粒度的涨跌停、停牌、盘口和成交数据，用于验证可成交性。

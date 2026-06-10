# 300316.SZ 晶盛机电单票量化研究系统

本项目围绕 A 股创业板 `300316.SZ` 晶盛机电建立日线级别研究闭环：数据适配、清洗、特征/标签、因子研究、walk-forward 模型、信号生成、扣成本回测、稳健性验证和最终评级。

核心原则：结论服从证据。如果样本外预测、扣成本回测、回撤控制或稳健性不足，报告必须给出 `B. 仅供观察` 或 `C. 不可用`，不能为了得到好看的收益曲线而调参、泄漏未来或伪造缺失数据。

## 当前结论

第二阶段复跑数据区间：`2018-01-02` 至 `2026-06-09`，共 2044 个交易日。

- 最终评级：`C. 不可用`
- 最新信号日期：`2026-06-09`
- 最新 `signal_label`：`risk_off`
- 最新 `target_position`：`0.0`
- 关键原因：5/20 日预测虽出现正 RankIC，但模型信号扣成本后总收益 `-1.22%`，显著弱于买入持有 `+231.15%`；风险分数为 `1.0`，最新信号触发风险过滤。
- 不升级评级原因：缺少指数、行业、财务、事件、产业链与盘口可成交性数据，且策略未能稳定转化为扣成本后的超额收益。

## 快速运行

```powershell
cd D:\Desktop\Quant\quant_300316
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -m unittest discover -s tests
python -B -m src.report --config config.yaml
python -B -m src.signal_generator --config config.yaml
```

最近一次验证：

- `python -B -m unittest discover -s tests`：`Ran 6 tests ... OK`
- `python -B -m src.report --config config.yaml`：成功生成全部报告
- `python -B -m src.signal_generator --config config.yaml`：输出最新 `risk_off` 信号，仓位 `0.0`

## 目录

```text
quant_300316/
  config.yaml
  data/
    raw/
      300316_daily.csv
      index_template.csv
      industry_template.csv
      financial_template.csv
      event_template.csv
      peer_template.csv
    processed/
  src/
    data_adapter.py
    data_cleaning.py
    feature_engineering.py
    label_builder.py
    factor_analysis.py
    feature_selection.py
    models.py
    walk_forward.py
    backtest.py
    signal_generator.py
    report.py
    utils.py
  reports/
    stock_research_report.md
    factor_report.md
    factor_deep_dive_report.md
    model_report.md
    backtest_report.md
    robustness_report.md
    final_decision_report.md
    daily_signal_template.csv
    trade_log.csv
    metrics_summary.csv
    equity_curve.png
  tests/
```

## 数据接口

默认使用项目内本地 CSV：`data/raw/300316_daily.csv`。

可选数据源和模板：

- 指数：`data/raw/index_template.csv`
- 行业：`data/raw/industry_template.csv`
- 财务：`data/raw/financial_template.csv`
- 事件：`data/raw/event_template.csv`
- 同行：`data/raw/peer_template.csv`

当前缺失且未纳入的数据：

- 沪深300、中证500、创业板指
- 申万或中信行业指数
- 光伏、半导体设备、硅片、碳化硅、设备订单等主题/产业链数据
- 财务公告日数据
- 业绩预告、定期报告、解禁、分红、股权激励、重大订单等事件数据
- 盘口、逐笔、真实停复牌和涨跌停可成交性数据

缺失数据不会被伪造，主流程会继续运行，并在报告中明确写明“缺失，因此未纳入”。

## 方法约束

- 所有模型使用 walk-forward，不随机打乱时间序列。
- 缺失值填充、标准化、特征筛选只在训练窗口 fit。
- `y_*` 和 `future_*` 列被排除在特征列之外。
- 财务数据必须按 `ann_date` 或 `announcement_date` 公告日对齐。
- 全样本因子报告只用于研究解释，不用于事后选因子再回测。
- 回测包含 commission、slippage、stamp tax，可配置 minimum fee。
- 信号是概率、风险和仓位参考，不是确定性投资建议。

## 主要报告

- [股票画像](reports/stock_research_report.md)
- [因子深挖](reports/factor_deep_dive_report.md)
- [模型报告](reports/model_report.md)
- [回测报告](reports/backtest_report.md)
- [稳健性报告](reports/robustness_report.md)
- [最终决策](reports/final_decision_report.md)

## 继续研究需要补充

优先补齐创业板指/沪深300/中证500、行业指数、同行标的、财务公告日、事件公告日和产业链价格数据。只有在这些数据加入后仍能保持样本外稳定、扣成本超额收益和可控回撤，才有资格从 `C. 不可用` 升级到 `B. 仅供观察` 或 `A. 可进入纸面交易`。

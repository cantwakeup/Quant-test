# 300316.SZ 单票量化预测与交易参考系统

本项目围绕晶盛机电 `300316.SZ` 建立日线级别的可复现研究闭环：数据适配、清洗、特征/标签、因子评价、walk-forward 模型、交易信号、含成本回测和报告输出。

核心原则：如果样本外预测和扣成本回测不稳定，报告必须给出“仅供观察”或“不可用”，不能为了得到好看的曲线强行调参。

## 目录

```text
quant_300316/
  README.md
  requirements.txt
  config.yaml
  data/
    raw/
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
  notebooks/
    01_data_check.ipynb
    02_factor_analysis.ipynb
    03_model_training.ipynb
    04_backtest_review.ipynb
  reports/
    factor_report.md
    model_report.md
    backtest_report.md
    daily_signal_template.csv
  tests/
    test_no_leakage.py
    test_feature_shift.py
    test_backtest_basic.py
```

## 快速运行

当前默认使用上层目录已有的本地 CSV：

```powershell
cd D:\Desktop\Quant\quant_300316
python -m src.report --config config.yaml
```

输出：

- `reports/factor_report.md`
- `reports/model_report.md`
- `reports/backtest_report.md`
- `reports/daily_signal_template.csv`
- `reports/equity_curve.png`
- `data/processed/*.csv`

生成最新收盘后信号：

```powershell
python -m src.signal_generator --config config.yaml
```

运行测试：

```powershell
python -m unittest discover -s tests
```

## 数据接口

`src/data_adapter.py` 提供统一接口：

- `CSVDataAdapter`
- `TushareAdapter`
- `AKShareAdapter`
- `EastmoneyAdapter`

默认 `config.yaml` 使用 CSV。若要接入 Tushare 或 AKShare，只需要修改：

```yaml
data:
  source: tushare
```

Tushare token 从环境变量读取：

```powershell
$env:TUSHARE_TOKEN="your-token"
```

财务数据必须包含 `ann_date` 或 `announcement_date`。系统使用 `merge_asof(..., direction="backward")` 按实际公告日对齐，禁止用报告期结束日提前泄漏。

## 研究方法

标签：

- `y_ret_1d/3d/5d/10d/20d`
- `y_up_5d/20d`
- `y_outperform_index_5d/20d`，在指数数据可用时生成
- `future_mdd_5d/20d`、`future_vol_5d/20d`、`y_crash_5d/20d`

候选因子：

- 趋势/动量
- 反转/超买超卖
- 成交量/流动性
- 波动率/风险
- 相对强弱
- 市场状态
- 基本面公告日特征
- 事件公告日特征

模型：

- 基线：买入持有、空仓、均线、动量
- 核心样本外模型：Ridge 回归、Logistic 分类
- scikit-learn 可用时可扩展 Ridge/Lasso/ElasticNet/RandomForest/Permutation Importance
- 当前实现保留纯 numpy 降级模型，避免环境不能导入 sklearn 时研究闭环中断

验证：

- expanding walk-forward
- purge/embargo
- 训练窗口内特征选择
- 标准化、缺失值填充只在训练集 fit
- 不打乱时间序列

## 交易信号

`daily_signal_template.csv` 字段包括：

- `date`
- `close`
- `pred_ret_5d`
- `pred_ret_20d`
- `prob_up_5d`
- `prob_up_20d`
- `risk_score`
- `signal_score`
- `signal_label`
- `target_position`
- `reason`
- `risk_warning`

`target_position` 是参数化仓位参考，不是确定性买卖指令。

## 风险与局限

- 单股票时间序列样本有限，容易受到阶段性行情影响。
- 无行业指数、市场宽度、融资融券、北向资金、产业链价格等数据时，相关因子不会被伪造。
- 本地 CSV 只能近似识别停牌和涨跌停，真实交易可成交性需要更细数据。
- 日频回测未模拟盘口冲击、排队、部分成交和盘中止损。
- 研究输出不构成投资建议。

# 假设驱动因子报告

| hypothesis           |   available_factors | mean_abs_ic   | best_factor        | enter_model   | reason                                                                 |
|:---------------------|--------------------:|:--------------|:-------------------|:--------------|:-----------------------------------------------------------------------|
| H1 中期趋势有效      |                   9 | 0.0391        | adx_14d            | True          | candidate only; final selection remains walk-forward train-window only |
| H2 高波动后风险加大  |                   7 | 0.0377        | gap_risk_20d       | True          | candidate only; final selection remains walk-forward train-window only |
| H3 放量突破/缩量回调 |                   4 | 0.0437        | turnover_ratio_20d | True          | candidate only; final selection remains walk-forward train-window only |
| H4 相对强弱更重要    |                   0 |               | nan                | False         | data unavailable or factors absent                                     |
| H5 财报周期影响趋势  |                   0 |               | nan                | False         | data unavailable or factors absent                                     |
| H6 事件冷却期降仓    |                   0 |               | nan                | False         | data unavailable or factors absent                                     |

单票时间序列 IC 不是横截面 IC；若与趋势基线高度重复，只能作为解释，不应叠加制造伪信号。

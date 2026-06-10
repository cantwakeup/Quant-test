# 元模型报告

| task                | model                |   observations |   brier |   accuracy |   positive_rate | adopt_as_primary   | reason                                                              |
|:--------------------|:---------------------|---------------:|--------:|-----------:|----------------:|:-------------------|:--------------------------------------------------------------------|
| good_trade_20d      | numpy_logistic_proxy |           1268 |  0.3259 |     0.5514 |          0.3572 | False              | proxy probability does not yet beat simple trend rules consistently |
| bad_trade_proxy_20d | numpy_logistic_proxy |           1268 |  0.4317 |     0.4393 |          0.4209 | False              | proxy probability does not yet beat simple trend rules consistently |
| up_after_cost_20d   | numpy_logistic_proxy |           1268 |  0.2924 |     0.521  |          0.5163 | False              | proxy probability does not yet beat simple trend rules consistently |

当前仅作为候选过滤器，复杂模型没有证明超过简单趋势基线。

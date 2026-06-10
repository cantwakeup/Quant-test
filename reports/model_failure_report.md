# 模型失败报告

| candidate            | type               | rank_ic   | direction_accuracy   | strategy_total_return   | max_drawdown   | adopt   | reason                                                                      |
|:---------------------|:-------------------|:----------|:---------------------|:------------------------|:---------------|:--------|:----------------------------------------------------------------------------|
| prediction_h1        | model_prediction   | 0.0295    | 0.4833               |                         |                | False   | prediction quality alone is insufficient; strategy conversion underperforms |
| prediction_h3        | model_prediction   | 0.0862    | 0.5089               |                         |                | False   | prediction quality alone is insufficient; strategy conversion underperforms |
| prediction_h5        | model_prediction   | 0.1005    | 0.5113               |                         |                | False   | prediction quality alone is insufficient; strategy conversion underperforms |
| prediction_h10       | model_prediction   | 0.1580    | 0.5454               |                         |                | False   | prediction quality alone is insufficient; strategy conversion underperforms |
| prediction_h20       | model_prediction   | 0.1439    | 0.5631               |                         |                | False   | prediction quality alone is insufficient; strategy conversion underperforms |
| prediction_h60       | model_prediction   | 0.1417    | 0.5822               |                         |                | False   | prediction quality alone is insufficient; strategy conversion underperforms |
| strategy             | backtest_portfolio |           |                      | -0.0122                 | -0.5255        | False   | baseline comparison                                                         |
| strategy_before_cost | backtest_portfolio |           |                      | 0.1447                  | -0.4804        | False   | baseline comparison                                                         |
| buy_hold_300316      | backtest_portfolio |           |                      | 2.3115                  | -0.7471        | False   | baseline comparison                                                         |
| ma_60_120            | trend_rule         |           |                      | 2.7462                  | -0.5257        | False   | candidate trend module; requires robustness and paper-trading validation    |
| momentum_20          | trend_rule         |           |                      | 2.5335                  | -0.6488        | False   | candidate trend module; requires robustness and paper-trading validation    |

失败模型不删除；它们用于约束最终评级。

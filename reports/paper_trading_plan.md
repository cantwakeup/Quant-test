# 纸面交易计划

## 状态

最终评级：**C. 不可用**

当前输出不是实盘建议。若评级不是 `A. 可进入纸面交易`，则只记录观察信号，不建立纸面持仓。

## 最新信号

| date                |   close |   pred_ret_5d |   pred_ret_20d |   prob_up_5d |   prob_up_20d | pred_excess_ret_5d   |   risk_score |   trend_score |   volume_score |   relative_strength_score |   fundamental_score |   event_score |   signal_score | signal_label   |   target_position |   stop_loss_reference |   take_profit_reference | invalidation_condition                                                               | reason                | risk_warning                             |   current_position |   max_position_allowed | action   | action_reason         | data_quality_flag     | manual_review_required   | next_trade_date   |   trailing_stop_reference |   meta_model_prob_good_trade_20d |   meta_model_prob_bad_trade_20d |   model_score | pred_excess_ret_20d   | expected_mfe_20d   | expected_mae_20d   |
|:--------------------|--------:|--------------:|---------------:|-------------:|--------------:|:---------------------|-------------:|--------------:|---------------:|--------------------------:|--------------------:|--------------:|---------------:|:---------------|------------------:|----------------------:|------------------------:|:-------------------------------------------------------------------------------------|:----------------------|:-----------------------------------------|-------------------:|-----------------------:|:---------|:----------------------|:----------------------|:-------------------------|:------------------|--------------------------:|---------------------------------:|--------------------------------:|--------------:|:----------------------|:-------------------|:-------------------|
| 2026-06-09 00:00:00 |   53.15 |        0.0056 |        -0.0842 |       0.4185 |        0.3203 |                      |            1 |        0.6863 |         0.5251 |                       0.5 |                 0.5 |             1 |        -1.1019 | risk_off       |                 0 |               42.5214 |                 65.9043 | close below stop reference, risk_score above 0.9, or signal flips to reduce/risk_off | risk filter triggered | recent volatility or drawdown is extreme |                  0 |                      0 | no_trade | risk filter triggered | missing_external_data | True                     | NaT               |                   42.5214 |                           0.3203 |                          0.6797 |       -1.1019 | <NA>                  | <NA>               | <NA>               |

## 纸面交易规则草案

- 开仓：仅当趋势模块、风险模块、相对强弱模块和模型/元模型同时支持，且 `manual_review_required=False`。
- 不开仓：`risk_off`、`no_trade`、高波动极端、接近涨跌停、外部数据缺失导致无法验证时。
- 减仓/清仓：跌破 stop reference、risk_score 超过 0.9、信号转为 reduce/risk_off。
- 仓位：0、0.25、0.5、0.75、1.0 分档；高风险状态自动限仓。
- 每日更新：收盘后重新运行 `python -B -m src.pipeline --config config.yaml --stage full`。

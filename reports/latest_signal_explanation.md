# 最新信号解释

- 最新交易日：2026-06-09 00:00:00
- 收盘价：53.15
- 趋势状态：trend_score=0.686251962398991
- 相对强弱状态：relative_strength_score=0.5；外部指数/行业/peer 缺失时为中性占位。
- 风险状态：risk_score=1.0，当前触发 `risk_off`。
- 财务/事件状态：当前财务和事件数据缺失，因此 fundamental_score/event_score 不能作为强证据。
- 模型预测：pred_ret_20d=-0.08420730526114152，prob_up_20d=0.32031881113093336。
- 最终 signal_label：risk_off
- target_position：0.0
- 为什么不是更高仓位：风险分数过高，且外部数据缺失，模型/规则没有足够扣成本交易优势。
- 为什么不是更低仓位：仓位已为 0。
- stop loss reference：42.521428571428565
- take profit reference：65.90428571428572
- invalidation condition：close below stop reference, risk_score above 0.9, or signal flips to reduce/risk_off
- data quality warning：指数、行业、财务、事件、peer 数据缺失，需要人工复核。

最终评级：C. 不可用。这不是实盘建议。

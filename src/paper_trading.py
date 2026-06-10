from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import to_markdown_table, write_markdown


def write_paper_trading_plan(path: Path, latest_signal: pd.DataFrame, final_rating: str) -> None:
    content = f"""# 纸面交易计划

## 状态

最终评级：**{final_rating}**

当前输出不是实盘建议。若评级不是 `A. 可进入纸面交易`，则只记录观察信号，不建立纸面持仓。

## 最新信号

{to_markdown_table(latest_signal)}

## 纸面交易规则草案

- 开仓：仅当趋势模块、风险模块、相对强弱模块和模型/元模型同时支持，且 `manual_review_required=False`。
- 不开仓：`risk_off`、`no_trade`、高波动极端、接近涨跌停、外部数据缺失导致无法验证时。
- 减仓/清仓：跌破 stop reference、risk_score 超过 0.9、信号转为 reduce/risk_off。
- 仓位：0、0.25、0.5、0.75、1.0 分档；高风险状态自动限仓。
- 每日更新：收盘后重新运行 `python -B -m src.pipeline --config config.yaml --stage full`。
"""
    write_markdown(path, content)

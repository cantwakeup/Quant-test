from __future__ import annotations

from pathlib import Path

from .utils import write_markdown


def write_execution_assumption_report(path: Path) -> None:
    content = """# 执行假设报告

- 信号生成：收盘后。
- 默认成交：下一交易日开盘价。
- 备选扰动：下一交易日收盘、延迟一天、滑点 2x/3x。
- 成本：commission、slippage、stamp tax、minimum fee。
- A 股限制：日频近似 T+1；若停牌/涨跌停字段可用则限制成交。
- 手数：当前回测以仓位比例近似，未模拟 100 股整数手和资金规模约束。
- 结论影响：缺少盘口和逐笔成交数据，因此任何评级不得等同实盘建议。
"""
    write_markdown(path, content)

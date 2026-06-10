from __future__ import annotations

import re
from typing import Dict


EVENT_PATTERNS = {
    "annual_report": r"年报|年度报告",
    "q1_report": r"一季报|第一季度",
    "semi_annual_report": r"半年报|半年度",
    "q3_report": r"三季报|第三季度",
    "earnings_forecast": r"业绩预告",
    "earnings_flash": r"业绩快报",
    "dividend": r"利润分配|分红|派息",
    "reduction": r"减持",
    "incentive": r"股权激励",
    "unlock": r"解禁|限售股上市",
    "buyback": r"回购",
    "major_order": r"重大合同|订单|中标",
    "regulatory_inquiry": r"问询函|监管",
    "investor_relations": r"投资者关系",
}


def classify_announcement_title(title: str) -> Dict[str, object]:
    text = str(title or "")
    matches = [name for name, pattern in EVENT_PATTERNS.items() if re.search(pattern, text)]
    if not matches:
        matches = ["generic_announcement"]
    return {
        "event_type": matches[0],
        "matched_types": matches,
        "confidence": 0.85 if matches[0] != "generic_announcement" else 0.35,
    }

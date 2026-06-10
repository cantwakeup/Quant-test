from __future__ import annotations

PEER_GROUPS = {
    "photovoltaic_equipment": {
        "300751.SZ": "迈为股份",
        "300724.SZ": "捷佳伟创",
        "688516.SH": "奥特维",
        "603396.SH": "金辰股份",
        "601908.SH": "京运通",
    },
    "semiconductor_equipment": {
        "002371.SZ": "北方华创",
        "688012.SH": "中微公司",
        "688082.SH": "盛美上海",
        "688072.SH": "拓荆科技",
        "688120.SH": "华海清科",
    },
    "advanced_manufacturing_growth": {
        "300316.SZ": "晶盛机电",
        "300751.SZ": "迈为股份",
        "300724.SZ": "捷佳伟创",
        "002371.SZ": "北方华创",
    },
}


def peer_group_table():
    rows = []
    for group, members in PEER_GROUPS.items():
        for symbol, name in members.items():
            rows.append({"group": group, "symbol": symbol, "name": name})
    return rows

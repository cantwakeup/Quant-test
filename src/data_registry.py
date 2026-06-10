from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


def describe_csv(path: Path, dataset: str, source: str, notes: str = "") -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "dataset": dataset,
        "path": str(path.as_posix()),
        "source": source,
        "notes": notes,
        "exists": path.exists(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if not path.exists():
        record.update({"rows": 0, "columns": [], "start_date": None, "end_date": None})
        return record
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        record.update({"rows": None, "columns": [], "error": repr(exc)})
        return record
    record["rows"] = int(len(df))
    record["columns"] = list(df.columns)
    if "date" in df.columns and len(df):
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        record["start_date"] = str(dates.min().date()) if len(dates) else None
        record["end_date"] = str(dates.max().date()) if len(dates) else None
    elif "ann_date" in df.columns and len(df):
        dates = pd.to_datetime(df["ann_date"], errors="coerce").dropna()
        record["start_date"] = str(dates.min().date()) if len(dates) else None
        record["end_date"] = str(dates.max().date()) if len(dates) else None
    else:
        record["start_date"] = None
        record["end_date"] = None
    return record


def build_manifest(base_dir: Path, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "project": "quant_300316",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base_dir),
        "datasets": list(records),
    }


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def manifest_to_markdown(manifest: Dict[str, Any]) -> str:
    rows = []
    for item in manifest.get("datasets", []):
        rows.append(
            {
                "dataset": item.get("dataset"),
                "exists": item.get("exists"),
                "rows": item.get("rows"),
                "start_date": item.get("start_date"),
                "end_date": item.get("end_date"),
                "source": item.get("source"),
                "notes": item.get("notes"),
            }
        )
    table = pd.DataFrame(rows)
    table_md = table.to_markdown(index=False) if not table.empty else "_No data registered._"
    return f"""# 数据清单

生成时间：{manifest.get('generated_at')}

{table_md}

缺失数据不会被伪造。模板文件只定义人工补数格式，不会进入模型特征。
"""

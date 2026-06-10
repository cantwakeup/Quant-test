from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import pandas as pd

from .data_cleaning import clean_ohlcv
from .utils import PROJECT_ROOT, resolve_path


DateLike = Union[str, date, datetime]


class DataAdapter(ABC):
    """Unified data adapter interface.

    Implementations may use local CSV, Tushare, AKShare, or another vendor.
    Every adapter returns normalized columns where possible:
    date, open, high, low, close, volume, amount, turnover, pct_change.
    """

    @abstractmethod
    def get_stock_daily(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        raise NotImplementedError

    def get_index_daily(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        adjust: str = "none",
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def get_industry_daily(
        self,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def get_theme_daily(
        self,
        theme: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def get_fundamentals(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def get_events(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def get_related_daily(
        self,
        symbols: Iterable[str],
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        adjust: str = "qfq",
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        for item in symbols:
            try:
                result[item] = self.get_stock_daily(item, start=start, end=end, adjust=adjust)
            except Exception:
                continue
        return result


class CSVDataAdapter(DataAdapter):
    def __init__(self, csv_config: Dict[str, object], base_dir: Optional[Path] = None):
        self.csv_config = csv_config or {}
        self.base_dir = base_dir or PROJECT_ROOT

    def _read_csv(self, path_value: Optional[str]) -> pd.DataFrame:
        path = resolve_path(path_value, self.base_dir)
        if path is None or not path.exists():
            return pd.DataFrame()
        data = pd.read_csv(path)
        # Treat header-only template files as intentionally missing data.
        return data if len(data) else pd.DataFrame(columns=data.columns)

    def get_stock_daily(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        symbol_paths = self.csv_config.get("stock_daily_paths", {}) or {}
        path_value = symbol_paths.get(symbol) or self.csv_config.get("stock_daily_path")
        data = self._read_csv(path_value)
        return clean_ohlcv(data, start=start, end=end)

    def get_index_daily(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        adjust: str = "none",
    ) -> pd.DataFrame:
        paths = self.csv_config.get("index_daily_paths", {}) or {}
        data = self._read_csv(paths.get(symbol))
        return clean_ohlcv(data, start=start, end=end) if not data.empty else data

    def get_industry_daily(
        self,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
    ) -> pd.DataFrame:
        data = self._read_csv(self.csv_config.get("industry_daily_path"))
        return clean_ohlcv(data, start=start, end=end) if not data.empty else data

    def get_theme_daily(
        self,
        theme: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
    ) -> pd.DataFrame:
        paths = self.csv_config.get("theme_daily_paths", {}) or {}
        data = self._read_csv(paths.get(theme))
        return clean_ohlcv(data, start=start, end=end) if not data.empty else data

    def get_fundamentals(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
    ) -> pd.DataFrame:
        data = self._read_csv(self.csv_config.get("fundamentals_path"))
        if data.empty:
            return data
        for col in ("ann_date", "announcement_date", "date", "report_date"):
            if col in data.columns:
                data[col] = pd.to_datetime(data[col])
        if start and "ann_date" in data.columns:
            data = data[data["ann_date"] >= pd.to_datetime(start)]
        if end and "ann_date" in data.columns:
            data = data[data["ann_date"] <= pd.to_datetime(end)]
        return data.sort_values([c for c in ("ann_date", "report_date") if c in data.columns])

    def get_events(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
    ) -> pd.DataFrame:
        data = self._read_csv(self.csv_config.get("events_path"))
        if data.empty:
            return data
        event_col = "event_date" if "event_date" in data.columns else "ann_date"
        data[event_col] = pd.to_datetime(data[event_col])
        if start:
            data = data[data[event_col] >= pd.to_datetime(start)]
        if end:
            data = data[data[event_col] <= pd.to_datetime(end)]
        if "event_date" not in data.columns:
            data = data.rename(columns={event_col: "event_date"})
        if "event_type" not in data.columns:
            data["event_type"] = "generic_event"
        return data.sort_values("event_date")

    def get_related_daily(
        self,
        symbols: Iterable[str],
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        adjust: str = "qfq",
    ) -> Dict[str, pd.DataFrame]:
        paths = self.csv_config.get("peer_daily_paths", {}) or {}
        result = {}
        for symbol in symbols:
            data = self._read_csv(paths.get(symbol))
            if not data.empty:
                result[symbol] = clean_ohlcv(data, start=start, end=end)
        return result


def compact_date(value: Optional[DateLike]) -> str:
    if value is None:
        return date.today().strftime("%Y%m%d")
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    cleaned = str(value).strip().replace("-", "")
    if not re.fullmatch(r"\d{8}", cleaned):
        raise ValueError(f"Date must be YYYY-MM-DD or YYYYMMDD: {value}")
    return cleaned


def infer_eastmoney_secid(symbol: str) -> str:
    raw = symbol.strip().upper()
    market = None
    code = raw
    if raw.endswith(".SH"):
        market, code = "1", raw[:-3]
    elif raw.endswith(".SZ"):
        market, code = "0", raw[:-3]
    code = re.sub(r"\D", "", code)
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError(f"Expected 6-digit A-share code, got {symbol}")
    if market is None:
        market = "1" if code.startswith(("5", "6", "9")) else "0"
    return f"{market}.{code}"


class EastmoneyAdapter(DataAdapter):
    """Network adapter useful when Tushare/AKShare are unavailable."""

    KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    def get_stock_daily(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        adjust_map = {"none": "0", "qfq": "1", "hfq": "2"}
        query = {
            "secid": infer_eastmoney_secid(symbol),
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",
            "fqt": adjust_map.get(adjust, "1"),
            "beg": compact_date(start),
            "end": compact_date(end),
        }
        url = f"{self.KLINE_URL}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        klines = (payload.get("data") or {}).get("klines") or []
        rows = []
        for item in klines:
            p = item.split(",")
            if len(p) >= 11:
                rows.append(
                    {
                        "date": p[0],
                        "open": p[1],
                        "close": p[2],
                        "high": p[3],
                        "low": p[4],
                        "volume": p[5],
                        "amount": p[6],
                        "amplitude": p[7],
                        "pct_change": p[8],
                        "change": p[9],
                        "turnover": p[10],
                    }
                )
        return clean_ohlcv(pd.DataFrame(rows), start=start, end=end)


class TushareAdapter(DataAdapter):
    def __init__(self, token_env: str = "TUSHARE_TOKEN"):
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeError(f"Tushare token not found in env var {token_env}")
        import tushare as ts  # type: ignore

        ts.set_token(token)
        self.ts = ts
        self.pro = ts.pro_api()

    def get_stock_daily(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        df = self.ts.pro_bar(
            ts_code=symbol,
            start_date=compact_date(start),
            end_date=compact_date(end),
            adj=adjust if adjust in {"qfq", "hfq"} else None,
            freq="D",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        return clean_ohlcv(df, start=start, end=end)

    def get_index_daily(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        adjust: str = "none",
    ) -> pd.DataFrame:
        df = self.ts.pro_bar(
            ts_code=symbol,
            start_date=compact_date(start),
            end_date=compact_date(end),
            asset="I",
            freq="D",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        return clean_ohlcv(df, start=start, end=end)

    def get_fundamentals(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
    ) -> pd.DataFrame:
        fields = (
            "ts_code,ann_date,end_date,report_type,basic_eps,total_revenue,"
            "revenue,n_income_attr_p,grossprofit_margin,netprofit_margin,roe,"
            "net_cash_flows_oper_act,inventory,accounts_receiv,contract_liab,cap_rese"
        )
        df = self.pro.fina_indicator(ts_code=symbol, start_date=compact_date(start), end_date=compact_date(end))
        if df is None or df.empty:
            return pd.DataFrame()
        return df.rename(columns={"ann_date": "ann_date", "end_date": "report_date"})


class AKShareAdapter(DataAdapter):
    def __init__(self):
        import akshare as ak  # type: ignore

        self.ak = ak

    @staticmethod
    def _ak_symbol(symbol: str) -> str:
        return re.sub(r"\D", "", symbol)

    def get_stock_daily(
        self,
        symbol: str,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        adjust_map = {"none": "", "qfq": "qfq", "hfq": "hfq"}
        df = self.ak.stock_zh_a_hist(
            symbol=self._ak_symbol(symbol),
            period="daily",
            start_date=compact_date(start),
            end_date=compact_date(end),
            adjust=adjust_map.get(adjust, "qfq"),
        )
        mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
        return clean_ohlcv(df.rename(columns=mapping), start=start, end=end)


def build_adapter(config: Dict[str, object]) -> DataAdapter:
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    source = str(data_cfg.get("source", "csv")).lower()
    base_dir = Path(config.get("_config_dir", PROJECT_ROOT)) if isinstance(config, dict) else PROJECT_ROOT
    if source == "csv":
        return CSVDataAdapter(data_cfg.get("csv", {}) or {}, base_dir=base_dir)
    if source == "eastmoney":
        return EastmoneyAdapter()
    if source == "tushare":
        token_env = (data_cfg.get("tushare", {}) or {}).get("token_env", "TUSHARE_TOKEN")
        return TushareAdapter(token_env=token_env)
    if source == "akshare":
        return AKShareAdapter()
    raise ValueError(f"Unsupported data source: {source}")

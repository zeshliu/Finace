"""免费 A 股数据源：AKShare 主源，BaoStock 日线备用源。"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd

LOGGER = logging.getLogger(__name__)


SPOT_RENAME = {
    "代码": "code",
    "名称": "name",
    "最新价": "price",
    "涨跌幅": "pct_change",
    "今开": "open",
    "昨收": "prev_close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "量比": "volume_ratio",
    "换手率": "turnover",
    "5分钟涨跌": "change_5m",
}

DAILY_RENAME = {
    "日期": "date",
    "股票代码": "code",
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

MINUTE_RENAME = {
    "时间": "datetime",
    "day": "datetime",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "均价": "average",
}

NUMERIC_COLUMNS = {
    "price", "pct_change", "open", "prev_close", "high", "low", "close", "volume", "amount",
    "volume_ratio", "turnover", "change_5m", "amplitude", "change", "average",
}

BAOSTOCK_LOCK = threading.Lock()


def normalize_code(value: str) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits[-6:].zfill(6)


def exchange_code(code: str) -> str:
    code = normalize_code(code)
    if code.startswith(("4", "8", "92")):
        return f"bj.{code}"
    return f"sh.{code}" if code.startswith(("5", "6")) else f"sz.{code}"


def normalize_frame(frame: pd.DataFrame, rename: dict[str, str], date_column: str | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.rename(columns=rename).copy()
    if "code" in result:
        result["code"] = result["code"].map(normalize_code)
    for column in NUMERIC_COLUMNS.intersection(result.columns):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if date_column and date_column in result:
        result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
        result = result.dropna(subset=[date_column]).sort_values(date_column).drop_duplicates(date_column, keep="last")
    return result.reset_index(drop=True)


def normalize_tencent_history(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    """腾讯第6列实际为成交量（手）；估算成交额供流动性筛选使用。"""
    required = {"date", "open", "close", "high", "low", "amount"}
    if frame is None or frame.empty or required.difference(frame.columns):
        raise ValueError(f"腾讯日线 {code} 字段不完整")
    result = frame.copy().rename(columns={"amount": "volume"})
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ("open", "close", "high", "low", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "open", "close", "high", "low", "volume"])
    result = result.sort_values("date").drop_duplicates("date", keep="last")
    typical_price = (result["open"] + result["close"] + result["high"] + result["low"]) / 4
    result["amount"] = result["volume"] * 100 * typical_price
    result["pct_change"] = result["close"].pct_change(fill_method=None) * 100
    result["code"] = normalize_code(code)
    if result.empty:
        raise ValueError(f"腾讯日线 {code} 标准化后为空")
    return result.reset_index(drop=True)


class MarketDataProvider:
    def __init__(
        self,
        retries: int = 3,
        retry_seconds: float = 2,
        timeout: float = 15,
        history_sources: list[str] | None = None,
    ):
        self.retries = max(1, int(retries))
        self.retry_seconds = max(0, float(retry_seconds))
        self.timeout = timeout
        self.history_sources = history_sources or ["tencent", "eastmoney", "baostock"]
        self._baostock_logged_in = False

    def _retry(self, label: str, operation: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                result = operation()
                if result is not None and not result.empty:
                    return result
                raise RuntimeError(f"{label} 返回空数据")
            except Exception as exc:  # 数据源异常类型不稳定，统一重试
                last_error = exc
                LOGGER.warning("%s 第 %s/%s 次失败: %s", label, attempt, self.retries, exc)
                if attempt < self.retries:
                    time.sleep(self.retry_seconds * attempt)
        raise RuntimeError(f"{label} 获取失败") from last_error

    def get_spot(self, min_rows: int = 0) -> pd.DataFrame:
        import akshare as ak

        required = {"code", "name", "price", "pct_change", "volume", "amount"}

        def validate(raw: pd.DataFrame) -> pd.DataFrame:
            normalized = normalize_frame(raw, SPOT_RENAME)
            missing = required.difference(normalized.columns)
            if missing:
                raise ValueError(f"A股快照缺少列: {', '.join(sorted(missing))}")
            if len(normalized) < int(min_rows):
                raise RuntimeError(f"A股快照仅返回 {len(normalized)} 行，低于完整性阈值 {min_rows}")
            return normalized

        try:
            return validate(self._retry("东方财富 A股快照", lambda: ak.stock_zh_a_spot_em()))
        except Exception as em_error:
            LOGGER.warning("东方财富快照失败，切换新浪快照: %s", em_error)
            return validate(self._retry("新浪 A股快照", lambda: ak.stock_zh_a_spot()))

    def get_history(self, code: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        """按配置依次尝试腾讯、东方财富和 BaoStock。日期格式 YYYYMMDD。"""
        code = normalize_code(code)
        errors: list[str] = []
        handlers = {
            "tencent": self._get_history_tencent,
            "eastmoney": self._get_history_eastmoney,
            "baostock": self._get_history_baostock,
        }
        for source in self.history_sources:
            handler = handlers.get(str(source).lower())
            if not handler:
                continue
            try:
                return handler(code, start_date, end_date, adjust)
            except Exception as exc:
                errors.append(f"{source}: {exc}")
                LOGGER.warning("%s 日线失败 %s: %s", source, code, exc)
        raise RuntimeError(f"{code} 所有日线源失败: {' | '.join(errors)}")

    def _get_history_tencent(self, code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        import akshare as ak

        symbol = exchange_code(code).replace(".", "")
        raw = self._retry(
            f"腾讯日线 {code}",
            lambda: ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                timeout=self.timeout,
            ),
        )
        return normalize_tencent_history(raw, code)

    def _get_history_eastmoney(self, code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        import akshare as ak

        raw = self._retry(
            f"东方财富日线 {code}",
            lambda: ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                timeout=self.timeout,
            ),
        )
        result = normalize_frame(raw, DAILY_RENAME, "date")
        daily_required = {"date", "open", "high", "low", "close", "volume", "amount"}
        if result.empty or daily_required.difference(result.columns):
            raise ValueError(f"东方财富日线 {code} 字段不完整")
        result["code"] = code
        return result

    def _get_history_baostock(self, code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        import baostock as bs

        # BaoStock 使用进程级会话；复用一次登录并串行查询少量回退股票。
        with BAOSTOCK_LOCK:
            if not self._baostock_logged_in:
                login = bs.login()
                if login.error_code != "0":
                    raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")
                self._baostock_logged_in = True
            fields = "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg,tradestatus"
            response = bs.query_history_k_data_plus(
                exchange_code(code),
                fields,
                start_date=datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d"),
                end_date=datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="2" if adjust == "qfq" else "3",
            )
            if response.error_code != "0":
                raise RuntimeError(response.error_msg)
            rows = []
            while response.next():
                rows.append(response.get_row_data())
            raw = pd.DataFrame(rows, columns=response.fields)
        if raw.empty:
            raise RuntimeError(f"BaoStock 日线 {code} 返回空数据")
        raw = raw.rename(columns={"turn": "turnover", "pctChg": "pct_change", "tradestatus": "trade_status"})
        raw["code"] = code
        result = normalize_frame(raw, {}, "date")
        if "trade_status" in result:
            result = result[result["trade_status"].astype(str) == "1"]
        return result.reset_index(drop=True)

    def close(self) -> None:
        if not self._baostock_logged_in:
            return
        with BAOSTOCK_LOCK:
            try:
                import baostock as bs

                bs.logout()
            finally:
                self._baostock_logged_in = False

    def get_intraday(self, code: str, trade_date: str | None = None, period: int = 5) -> pd.DataFrame:
        import akshare as ak

        day = trade_date or datetime.now().strftime("%Y-%m-%d")
        start = f"{day} 09:30:00"
        end = f"{day} 15:00:00"
        try:
            raw = self._retry(
                f"东方财富分钟线 {code}",
                lambda: ak.stock_zh_a_hist_min_em(
                    symbol=normalize_code(code),
                    start_date=start,
                    end_date=end,
                    period=str(period),
                    adjust="",
                ),
            )
        except Exception as em_error:
            LOGGER.warning("东方财富分钟线失败，切换新浪 %s: %s", code, em_error)
            raw = self._retry(
                f"新浪分钟线 {code}",
                lambda: ak.stock_zh_a_minute(symbol=exchange_code(code).replace(".", ""), period=str(period), adjust=""),
            )
        result = normalize_frame(raw, MINUTE_RENAME, "datetime")
        minute_required = {"datetime", "open", "high", "low", "close", "volume", "amount"}
        if minute_required.difference(result.columns):
            raise ValueError(f"{code} 分钟线字段不完整")
        if not result.empty:
            result = result[result["datetime"].dt.strftime("%Y-%m-%d") == day]
        if result.empty:
            raise RuntimeError(f"{code} 在 {day} 没有有效分钟线")
        return result.reset_index(drop=True)


def history_request_range(cached: pd.DataFrame, end: datetime, lookback_days: int) -> tuple[str, str]:
    """有缓存时只补最近日期并留 5 天重叠，无缓存时拉取完整窗口。"""
    if cached is not None and not cached.empty and "date" in cached:
        latest = pd.to_datetime(cached["date"], errors="coerce").max()
        start = latest.to_pydatetime() - timedelta(days=5) if pd.notna(latest) else end - timedelta(days=lookback_days)
    else:
        start = end - timedelta(days=lookback_days)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

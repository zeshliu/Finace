"""免费 A 股数据源：腾讯、东方财富主源，新浪财经和 BaoStock 备用。"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd

LOGGER = logging.getLogger(__name__)

_REQUEST_TIMEOUT_LOCK = threading.Lock()
_REQUEST_TIMEOUT_SECONDS = 15.0
_REQUEST_TIMEOUT_INSTALLED = False
_REQUESTS_ORIGINAL_REQUEST = None


def _request_with_default_timeout(session, method, url, **kwargs):
    """为 AKShare 内部未声明 timeout 的 requests 调用补上读取超时。"""
    kwargs.setdefault("timeout", _REQUEST_TIMEOUT_SECONDS)
    return _REQUESTS_ORIGINAL_REQUEST(session, method, url, **kwargs)


def install_requests_default_timeout(timeout: float) -> None:
    """进程级安装一次超时兜底，避免第三方行情请求永久阻塞线程池。"""
    import requests

    global _REQUEST_TIMEOUT_SECONDS, _REQUEST_TIMEOUT_INSTALLED, _REQUESTS_ORIGINAL_REQUEST
    with _REQUEST_TIMEOUT_LOCK:
        _REQUEST_TIMEOUT_SECONDS = max(1.0, float(timeout))
        if _REQUEST_TIMEOUT_INSTALLED:
            return
        _REQUESTS_ORIGINAL_REQUEST = requests.sessions.Session.request
        requests.sessions.Session.request = _request_with_default_timeout
        _REQUEST_TIMEOUT_INSTALLED = True


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
    "zxj": "price",
    "zdf": "pct_change",
    "turnover": "amount",
}

ETF_SPOT_RENAME = {
    **SPOT_RENAME,
    "IOPV实时估值": "iopv",
    "基金折价率": "discount_rate",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "昨收": "prev_close",
    "数据日期": "data_date",
    "更新时间": "updated_at",
}

ETF_CATALOG_RENAME = {
    "基金代码": "code",
    "基金简称": "name",
    "类型": "fund_type",
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
    "volume_ratio", "turnover", "change_5m", "amplitude", "change", "average", "iopv", "discount_rate",
}

BAOSTOCK_LOCK = threading.Lock()


def normalize_code(value: str) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits[-6:].zfill(6)


def board_for_code(value: str) -> str:
    """按交易所代码规则识别A股板块。"""
    code = normalize_code(value)
    if code.startswith(("600", "601", "603", "605")):
        return "sh_main"
    if code.startswith(("000", "001", "002", "003")):
        return "sz_main"
    if code.startswith(("688", "689")):
        return "sh_star"
    if code.startswith(("300", "301")):
        return "sz_growth"
    if code.startswith(("4", "8", "92")):
        return "beijing"
    return "other"


def exchange_code(code: str) -> str:
    code = normalize_code(code)
    if code.startswith(("4", "8", "92")):
        return f"bj.{code}"
    return f"sh.{code}" if code.startswith(("5", "6")) else f"sz.{code}"


def normalize_industry_name(value: str) -> str:
    """清理行业代码，并将过长的证监会行业名称转换为列表友好的名称。"""
    text = str(value or "").strip()
    if len(text) >= 2 and text[0].isalpha():
        index = 1
        while index < len(text) and text[index].isdigit():
            index += 1
        if index > 1:
            text = text[index:]
    aliases = {
        "计算机、通信和其他电子设备制造业": "电子设备",
        "电力、热力生产和供应业": "电力",
        "软件和信息技术服务业": "软件服务",
        "互联网和相关服务": "互联网",
        "货币金融服务": "银行",
        "资本市场服务": "证券",
        "医药制造业": "医药制造",
        "汽车制造业": "汽车",
    }
    if text in aliases:
        return aliases[text]
    if text.endswith("制造业") and len(text) > 3:
        return text[:-1]
    return text or "未分类"


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


def normalize_sina_history(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    """标准化新浪日线；成交量单位为股，成交额单位为元。"""
    result = normalize_frame(frame, {}, "date")
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    if result.empty or required.difference(result.columns):
        raise ValueError(f"新浪日线 {normalize_code(code)} 字段不完整")
    result["pct_change"] = result["close"].pct_change(fill_method=None) * 100
    result["code"] = normalize_code(code)
    return result.reset_index(drop=True)


class MarketDataProvider:
    def __init__(
        self,
        retries: int = 3,
        retry_seconds: float = 2,
        timeout: float = 15,
        spot_sources: list[str] | None = None,
        history_sources: list[str] | None = None,
        intraday_sources: list[str] | None = None,
        sina_history_interval: float = 0.25,
    ):
        self.retries = max(1, int(retries))
        self.retry_seconds = max(0, float(retry_seconds))
        self.timeout = timeout
        install_requests_default_timeout(timeout)
        self.spot_sources = spot_sources or ["sina", "eastmoney", "tencent"]
        self.history_sources = history_sources or ["tencent", "eastmoney", "sina", "baostock"]
        self.intraday_sources = intraday_sources or ["sina", "eastmoney"]
        self.sina_history_interval = max(0, float(sina_history_interval))
        self._sina_history_lock = threading.Lock()
        self._sina_history_last_started = 0.0
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

        handlers = {
            "sina": ("新浪 A股快照", ak.stock_zh_a_spot),
            "eastmoney": ("东方财富 A股快照", ak.stock_zh_a_spot_em),
            "tencent": ("腾讯 A股快照", ak.stock_zh_a_spot_tx),
        }
        errors: list[str] = []
        for source in self.spot_sources:
            normalized_source = str(source).lower()
            if normalized_source not in handlers:
                continue
            label, operation = handlers[normalized_source]
            try:
                return validate(self._retry(label, operation))
            except Exception as exc:
                errors.append(f"{normalized_source}: {exc}")
                LOGGER.warning("%s失败，尝试下一个快照源: %s", label, exc)
        raise RuntimeError(f"所有A股快照源失败: {' | '.join(errors)}")

    def get_etf_spot(self, min_rows: int = 0) -> pd.DataFrame:
        """获取 ETF 实时行情并合并基金类型，供 T+0 品种识别使用。"""
        import akshare as ak

        raw_spot = self._retry("东方财富 ETF 快照", ak.fund_etf_spot_em)
        spot = normalize_frame(raw_spot, ETF_SPOT_RENAME)
        required = {"code", "name", "price", "pct_change", "volume", "amount"}
        missing = required.difference(spot.columns)
        if missing:
            raise ValueError(f"ETF 快照缺少列: {', '.join(sorted(missing))}")
        if len(spot) < int(min_rows):
            raise RuntimeError(f"ETF 快照仅返回 {len(spot)} 行，低于完整性阈值 {min_rows}")

        raw_catalog = self._retry("东方财富 ETF 基金类型", ak.fund_etf_fund_daily_em)
        catalog = normalize_frame(raw_catalog, ETF_CATALOG_RENAME)
        if "code" not in catalog or "fund_type" not in catalog:
            raise ValueError("ETF 基金类型列表字段不完整")
        type_map = catalog.drop_duplicates("code", keep="first").set_index("code")["fund_type"]
        spot["fund_type"] = spot["code"].map(type_map).fillna("")
        return spot.reset_index(drop=True)

    def get_history(self, code: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        """按配置依次尝试新浪及免费备用源。日期格式 YYYYMMDD。"""
        code = normalize_code(code)
        errors: list[str] = []
        handlers = {
            "sina": self._get_history_sina,
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

    def get_etf_history(self, code: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        """ETF 日线：东方财富主源，新浪作为免费备用源。"""
        import akshare as ak

        code = normalize_code(code)
        errors: list[str] = []
        try:
            raw = self._retry(
                f"东方财富 ETF 日线 {code}",
                lambda: ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                ),
            )
            result = normalize_frame(raw, DAILY_RENAME, "date")
            required = {"date", "open", "high", "low", "close", "volume", "amount"}
            if result.empty or required.difference(result.columns):
                raise ValueError(f"东方财富 ETF 日线 {code} 字段不完整")
            result["code"] = code
            return result.reset_index(drop=True)
        except Exception as exc:
            errors.append(f"eastmoney: {exc}")
            LOGGER.warning("东方财富 ETF 日线失败 %s: %s", code, exc)

        try:
            symbol = exchange_code(code).replace(".", "")
            raw = self._retry(f"新浪 ETF 日线 {code}", lambda: ak.fund_etf_hist_sina(symbol=symbol))
            result = normalize_sina_history(raw, code)
            start = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
            end = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")
            if pd.notna(start):
                result = result[result["date"] >= start]
            if pd.notna(end):
                result = result[result["date"] <= end]
            if result.empty:
                raise ValueError(f"新浪 ETF 日线 {code} 日期范围内为空")
            return result.reset_index(drop=True)
        except Exception as exc:
            errors.append(f"sina: {exc}")
            LOGGER.warning("新浪 ETF 日线失败 %s: %s", code, exc)
        raise RuntimeError(f"{code} 所有 ETF 日线源失败: {' | '.join(errors)}")

    def get_industry(self, code: str) -> str:
        """获取东方财富细分行业；单只请求便于只查询最终候选。"""
        import requests

        code = normalize_code(code)
        market = "1" if code.startswith(("5", "6")) else "0"

        def fetch() -> pd.DataFrame:
            errors: list[str] = []
            params = {
                "fltt": "2",
                "invt": "2",
                "fields": "f57,f58,f127",
                "secid": f"{market}.{code}",
            }
            for host in ("push2delay.eastmoney.com", "push2.eastmoney.com"):
                try:
                    response = requests.get(
                        f"https://{host}/api/qt/stock/get",
                        params=params,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    data = response.json().get("data") or {}
                    industry = normalize_industry_name(data.get("f127"))
                    if industry == "未分类":
                        raise ValueError(f"{code} 未返回行业")
                    return pd.DataFrame([{"industry": industry}])
                except Exception as exc:
                    errors.append(f"{host}: {exc}")
            raise RuntimeError(" | ".join(errors))

        result = self._retry(f"东方财富行业 {code}", fetch)
        return str(result.iloc[0]["industry"])

    def _ensure_baostock_login_locked(self) -> None:
        import baostock as bs

        if self._baostock_logged_in:
            return
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")
        self._baostock_logged_in = True

    def get_industry_map_baostock(self) -> dict[str, str]:
        """一次查询全市场证监会行业，作为细分行业接口的免费备用源。"""
        import baostock as bs

        with BAOSTOCK_LOCK:
            self._ensure_baostock_login_locked()
            response = bs.query_stock_industry()
            if response.error_code != "0":
                raise RuntimeError(response.error_msg)
            mapping: dict[str, str] = {}
            while response.next():
                row = dict(zip(response.fields, response.get_row_data()))
                code = normalize_code(row.get("code", ""))
                industry = normalize_industry_name(row.get("industry", ""))
                if code and industry != "未分类":
                    mapping[code] = industry
        if not mapping:
            raise RuntimeError("BaoStock 行业分类返回空数据")
        return mapping

    def _throttle_sina_history(self) -> None:
        if self.sina_history_interval <= 0:
            return
        with self._sina_history_lock:
            elapsed = time.monotonic() - self._sina_history_last_started
            remaining = self.sina_history_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._sina_history_last_started = time.monotonic()

    def _get_history_sina(self, code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        import akshare as ak

        def fetch() -> pd.DataFrame:
            self._throttle_sina_history()
            return ak.stock_zh_a_daily(
                symbol=exchange_code(code).replace(".", ""),
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )

        raw = self._retry(f"新浪日线 {code}", fetch)
        return normalize_sina_history(raw, code)

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
            self._ensure_baostock_login_locked()
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
        normalized_code = normalize_code(code)
        handlers = {
            "sina": (
                "新浪分钟线",
                lambda: ak.stock_zh_a_minute(
                    symbol=exchange_code(normalized_code).replace(".", ""),
                    period=str(period),
                    adjust="",
                ),
            ),
            "eastmoney": (
                "东方财富分钟线",
                lambda: ak.stock_zh_a_hist_min_em(
                    symbol=normalized_code,
                    start_date=start,
                    end_date=end,
                    period=str(period),
                    adjust="",
                ),
            ),
        }
        errors: list[str] = []
        minute_required = {"datetime", "open", "high", "low", "close", "volume", "amount"}
        for source in self.intraday_sources:
            normalized_source = str(source).lower()
            if normalized_source not in handlers:
                continue
            label, operation = handlers[normalized_source]
            try:
                raw = self._retry(f"{label} {normalized_code}", operation)
                result = normalize_frame(raw, MINUTE_RENAME, "datetime")
                if minute_required.difference(result.columns):
                    raise ValueError(f"{normalized_code} 分钟线字段不完整")
                result = result[result["datetime"].dt.strftime("%Y-%m-%d") == day]
                if result.empty:
                    raise RuntimeError(f"{normalized_code} 在 {day} 没有有效分钟线")
                return result.reset_index(drop=True)
            except Exception as exc:
                errors.append(f"{normalized_source}: {exc}")
                LOGGER.warning("%s %s失败，尝试下一个分钟线源: %s", label, normalized_code, exc)
        raise RuntimeError(f"{normalized_code} 所有分钟线源失败: {' | '.join(errors)}")


def history_request_range(cached: pd.DataFrame, end: datetime, lookback_days: int) -> tuple[str, str]:
    """有缓存时只补最近日期并留 5 天重叠，无缓存时拉取完整窗口。"""
    if cached is not None and not cached.empty and "date" in cached:
        latest = pd.to_datetime(cached["date"], errors="coerce").max()
        start = latest.to_pydatetime() - timedelta(days=5) if pd.notna(latest) else end - timedelta(days=lookback_days)
    else:
        start = end - timedelta(days=lookback_days)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

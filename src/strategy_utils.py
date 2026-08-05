"""策略共享的基础筛选和详情序列构建。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .providers import board_for_code, normalize_code


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return default if not np.isfinite(result) else result
    except (TypeError, ValueError):
        return default


def limit_ratio_for_code(code: str, config: dict) -> float:
    ratios = config["screening"]["near_limit_ratio"]
    code = normalize_code(code)
    if code.startswith(("300", "301", "688", "689")):
        return float(ratios["growth"])
    if code.startswith(("4", "8", "92")):
        return float(ratios["beijing"])
    return float(ratios["main"])


def is_st_name(name: str) -> bool:
    normalized = str(name or "").upper().replace(" ", "")
    return normalized.startswith("ST") or normalized.startswith("*ST")


def passes_basic_filters(
    spot: dict | pd.Series,
    history: pd.DataFrame,
    config: dict,
    *,
    apply_return_limits: bool = True,
) -> tuple[bool, list[str]]:
    """基础范围过滤；可按策略关闭涨跌幅相关限制，价格边界为闭区间。"""
    reasons: list[str] = []
    screening = config["screening"]
    price = _number(spot.get("price"), np.nan)
    name = str(spot.get("name", ""))
    code = normalize_code(spot.get("code", ""))

    allowed_boards = set(screening.get("allowed_boards") or [])
    if allowed_boards and board_for_code(code) not in allowed_boards:
        reasons.append("不属于沪市主板或深市主板")
    if not np.isfinite(price) or not (float(screening["price_min"]) <= price <= float(screening["price_max"])):
        reasons.append("价格不在设定区间")
    if screening.get("exclude_st", True) and is_st_name(name):
        reasons.append("ST股票")
    if screening.get("exclude_suspended", True) and _number(spot.get("volume"), 0) <= 0:
        reasons.append("停牌或无成交")
    if history is None or history.empty:
        reasons.append("无历史数据")
        return False, reasons

    valid_history = history.dropna(subset=["close"]) if "close" in history else pd.DataFrame()
    min_listing_days = int(screening.get("min_listing_trading_days", 0) or 0)
    if min_listing_days > 0 and len(valid_history) < min_listing_days:
        reasons.append("上市交易日不足")

    min_avg_amount = float(screening.get("min_avg_amount_20", 0) or 0)
    if min_avg_amount > 0:
        if "amount" not in valid_history:
            reasons.append("缺少成交额数据")
        else:
            avg_amount = pd.to_numeric(valid_history["amount"], errors="coerce").tail(20).mean()
            if pd.isna(avg_amount) or avg_amount < min_avg_amount:
                reasons.append("近20日平均成交额不足")

    if apply_return_limits and len(valid_history) >= 2:
        previous_close = _number(valid_history["close"].iloc[-2], 0)
        current_close = _number(valid_history["close"].iloc[-1], price)
        # 日线已包含今天时，最新收盘应与快照几乎一致；否则盘中用快照价参与计算。
        if previous_close > 0 and abs(current_close - price) / max(price, 0.01) > 0.001:
            current_close = price
        day_return = current_close / previous_close - 1 if previous_close else 0
        if day_return >= limit_ratio_for_code(code, config):
            reasons.append("当天涨停或接近涨停")

    closes = pd.to_numeric(valid_history["close"], errors="coerce").dropna()
    max_3d_return = screening.get("max_3d_return")
    if apply_return_limits and max_3d_return is not None and len(closes) >= 4:
        if abs(closes.iloc[-1] - price) / max(price, 0.01) <= 0.001:
            return_3d = closes.iloc[-1] / closes.iloc[-4] - 1
        else:
            return_3d = price / closes.iloc[-3] - 1
        if return_3d > float(max_3d_return):
            reasons.append("近3日累计涨幅过大")
    return not reasons, reasons


def finite_or_none(value: Any, digits: int = 4):
    try:
        number = float(value)
        return round(number, digits) if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def build_chart_data(history: pd.DataFrame, days: int = 120) -> list[dict]:
    if history is None or history.empty:
        return []
    columns = [
        "open", "close", "low", "high", "volume", "ma5", "ma10", "ma20", "ma60",
        "dif", "dea", "macd_hist", "k", "d", "j", "rsi6", "boll_upper", "boll_mid", "boll_lower",
        "atr14", "atr_ma20", "amplitude_pct",
    ]
    records: list[dict] = []
    for _, row in history.tail(days).iterrows():
        raw_date = row.get("date")
        date = pd.to_datetime(raw_date, errors="coerce")
        item = {"date": date.strftime("%Y-%m-%d") if pd.notna(date) else str(raw_date)}
        for column in columns:
            item[column] = finite_or_none(row.get(column))
        records.append(item)
    return records

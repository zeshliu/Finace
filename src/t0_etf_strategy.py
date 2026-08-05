"""普通 A 股账户可交易的 T+0 ETF 分类、筛选与评分。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .providers import normalize_code
from .strategy_utils import finite_or_none


MONEY_KEYWORDS = (
    "货币ETF", "交易货币", "财富宝ETF", "保证金", "收益宝", "添富快线", "现金添富", "华宝添益", "银华日利", "建信添益",
)
CROSS_BORDER_KEYWORDS = (
    "港股", "香港", "恒生", "中概", "纳指", "纳斯达克", "标普", "日经", "东证", "德国", "法国", "美国", "道琼斯",
    "东南亚", "亚太", "沙特", "巴西", "韩国", "中韩", "海外", "全球", "印度", "越南", "新加坡",
)
DOMESTIC_STOCK_MARKERS = ("黄金股", "现金流")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def classify_t0_etf(name: str, fund_type: str = "", code: str = "") -> str | None:
    """按基金类型为主、名称为辅识别交易所允许当日回转的 ETF。"""
    text = str(name or "").replace(" ", "")
    type_text = str(fund_type or "").replace(" ", "")
    if any(marker in text for marker in DOMESTIC_STOCK_MARKERS):
        return None
    if any(keyword in text for keyword in MONEY_KEYWORDS):
        return "货币ETF"
    if "固收" in type_text or any(keyword in text for keyword in ("国债ETF", "地方债ETF", "政金债ETF", "信用债ETF", "公司债ETF", "城投债ETF", "短融ETF", "可转债ETF", "科创债ETF")):
        return "债券ETF"
    if "海外股票" in type_text or any(keyword in text for keyword in CROSS_BORDER_KEYWORDS):
        return "跨境ETF"
    if "其他" in type_text:
        if "金ETF" in text or "黄金ETF" in text or "上海金ETF" in text:
            return "黄金ETF"
        return "商品ETF"
    return None


def _states(history: pd.DataFrame) -> tuple[str, str, str]:
    latest = history.iloc[-1]
    previous = history.iloc[-2]
    if latest["dif"] > latest["dea"] and latest["macd_hist"] > previous["macd_hist"]:
        macd_state = "多头扩张" if latest["macd_hist"] >= 0 else "零轴下方改善"
    elif latest["dif"] > latest["dea"]:
        macd_state = "多头收敛"
    else:
        macd_state = "偏弱"

    golden_cross = latest["k"] > latest["d"] and previous["k"] <= previous["d"]
    if golden_cross:
        kdj_state = "金叉向上"
    elif latest["k"] > latest["d"] and latest["j"] > previous["j"]:
        kdj_state = "多头向上"
    elif latest["j"] >= 90:
        kdj_state = "高位钝化"
    else:
        kdj_state = "偏弱"

    if latest["close"] >= latest["ma5"] >= latest["ma10"] >= latest["ma20"]:
        ma_state = "多头排列"
    elif latest["close"] >= latest["ma5"] >= latest["ma10"]:
        ma_state = "站上MA5/MA10"
    elif latest["close"] >= latest["ma5"]:
        ma_state = "均线修复"
    else:
        ma_state = "均线下方"
    return macd_state, kdj_state, ma_state


def score_t0_etf(spot: dict | pd.Series, raw_history: pd.DataFrame, config: dict) -> dict | None:
    settings = config["t0_etf"]
    history = add_indicators(raw_history)
    if len(history) < 40:
        return None
    latest, previous = history.iloc[-1], history.iloc[-2]
    last20 = history.tail(20)
    avg_amount_20 = _number(last20["amount"].mean(), np.nan)
    avg_amplitude_20 = _number(last20["amplitude_pct"].mean(), np.nan)
    current_amount = _number(spot.get("amount"), _number(latest.get("amount")))
    if not np.isfinite(avg_amount_20) or avg_amount_20 < float(settings["min_avg_amount_20"]):
        return None
    if not np.isfinite(avg_amplitude_20) or avg_amplitude_20 < float(settings["min_avg_amplitude_20"]):
        return None
    if current_amount < float(settings.get("min_current_amount", 0)) or _number(spot.get("volume"), _number(latest.get("volume"))) <= 0:
        return None

    fund_category = classify_t0_etf(spot.get("name", ""), spot.get("fund_type", ""), spot.get("code", ""))
    if not fund_category:
        return None

    macd_score = 0.0
    reasons: list[str] = []
    risks: list[str] = []
    if latest["dif"] > latest["dea"]:
        macd_score += 8
    if latest["macd_hist"] > 0:
        macd_score += 5
    if latest["macd_hist"] > previous["macd_hist"]:
        macd_score += 4
    if latest["dif"] > previous["dif"]:
        macd_score += 3
    if macd_score >= 12:
        reasons.append("MACD动能改善")

    kdj_score = 0.0
    golden_cross = latest["k"] > latest["d"] and previous["k"] <= previous["d"]
    if latest["k"] > latest["d"]:
        kdj_score += 8
    if golden_cross:
        kdj_score += 5
        reasons.append("KDJ金叉")
    if latest["j"] > previous["j"]:
        kdj_score += 4
    if 20 <= latest["k"] <= 80:
        kdj_score += 3

    ma_score = 0.0
    if latest["close"] >= latest["ma5"]:
        ma_score += 5
    if latest["ma5"] >= latest["ma10"]:
        ma_score += 5
    if latest["ma10"] >= latest["ma20"]:
        ma_score += 5
    if latest["ma5"] > previous["ma5"]:
        ma_score += 3
    if latest["close"] >= latest["ma20"]:
        ma_score += 2
    if ma_score >= 13:
        reasons.append("均线结构偏强")

    atr14 = _number(latest.get("atr14"), np.nan)
    atr_ma20 = _number(latest.get("atr_ma20"), np.nan)
    atr_score = 0.0
    if np.isfinite(atr14) and np.isfinite(atr_ma20) and atr14 > atr_ma20:
        atr_score += 6
        reasons.append("ATR高于20日均值")
    atr_values = pd.to_numeric(history["atr14"], errors="coerce").dropna().tail(3)
    atr_rising = len(atr_values) == 3 and bool((np.diff(atr_values.to_numpy()) > 0).all())
    if atr_rising:
        atr_score += 5
        reasons.append("ATR连续上升")
    if avg_amplitude_20 >= 1.2:
        atr_score += 4
    elif avg_amplitude_20 >= 1.0:
        atr_score += 2
    elif avg_amplitude_20 >= float(settings["min_avg_amplitude_20"]):
        atr_score -= 2
        risks.append("近20日平均振幅接近筛选下限")
    else:
        atr_score -= 5
        risks.append("近20日平均振幅过低")
    abnormal_multiplier = float(settings.get("atr_abnormal_multiplier", 2.0))
    if np.isfinite(atr14) and np.isfinite(atr_ma20) and atr_ma20 > 0 and atr14 >= atr_ma20 * abnormal_multiplier:
        atr_score -= 4
        risks.append("ATR异常放大，高波动风险")
    atr_score = max(0.0, min(15.0, atr_score))

    volume_ratio = _number(spot.get("volume_ratio"), 0)
    if volume_ratio <= 0:
        prior_volume = pd.to_numeric(history["volume"], errors="coerce").iloc[-6:-1].mean()
        volume_ratio = _number(latest["volume"] / prior_volume if prior_volume else 0)
    volume_score = 0.0
    if 1.0 <= volume_ratio <= 2.5:
        volume_score += 8
        reasons.append("量比温和放大")
    elif volume_ratio > 2.5:
        volume_score += 5
        risks.append("量比偏高，注意冲高回落")
    elif volume_ratio >= 0.8:
        volume_score += 4
    if avg_amount_20 >= 100_000_000:
        volume_score += 4
    else:
        volume_score += 2
    if current_amount >= avg_amount_20:
        volume_score += 3
    volume_score = min(15.0, volume_score)

    rsi6 = _number(latest.get("rsi6"), np.nan)
    if 45 <= rsi6 <= 70:
        rsi_score = 10.0
        reasons.append("RSI处于强势区间")
    elif 35 <= rsi6 <= 80:
        rsi_score = 7.0
    elif 25 <= rsi6 <= 85:
        rsi_score = 4.0
    else:
        rsi_score = 1.0
    if rsi6 > 80:
        risks.append("RSI偏高，短线可能过热")
    elif rsi6 < 25:
        risks.append("RSI偏低，弱势风险仍在")

    components = {
        "MACD": round(macd_score, 1),
        "KDJ": round(kdj_score, 1),
        "均线": round(ma_score, 1),
        "ATR": round(atr_score, 1),
        "成交量与量比": round(volume_score, 1),
        "RSI": round(rsi_score, 1),
    }
    score = round(max(0.0, min(100.0, sum(components.values()))), 1)
    macd_state, kdj_state, ma_state = _states(history)
    price = _number(spot.get("price"), _number(latest.get("close")))
    return {
        "code": normalize_code(spot.get("code", "")),
        "name": str(spot.get("name", "")),
        "category": fund_category,
        "price": finite_or_none(price, 3),
        "change_pct": finite_or_none(spot.get("pct_change"), 2),
        "score": score,
        "score_components": components,
        "macd_state": macd_state,
        "kdj_state": kdj_state,
        "ma_state": ma_state,
        "rsi6": finite_or_none(rsi6, 2),
        "atr14": finite_or_none(atr14, 4),
        "atr_pct": finite_or_none(atr14 / price * 100 if price else None, 2),
        "amount": finite_or_none(current_amount, 0),
        "avg_amount_20": finite_or_none(avg_amount_20, 0),
        "avg_amplitude_20": finite_or_none(avg_amplitude_20, 2),
        "volume_ratio": finite_or_none(volume_ratio, 2),
        "reasons": reasons or ["多项技术指标达到观察阈值"],
        "risks": risks,
    }


def scan_t0_etfs(spot_frame: pd.DataFrame, histories: dict[str, pd.DataFrame], config: dict) -> list[dict]:
    candidates: list[dict] = []
    threshold = float(config["t0_etf"]["score_threshold"])
    for _, row in spot_frame.iterrows():
        code = normalize_code(row.get("code", ""))
        history = histories.get(code)
        if history is None or history.empty:
            continue
        try:
            item = score_t0_etf(row, history, config)
            if item and item["score"] >= threshold:
                candidates.append(item)
        except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError):
            continue
    return sorted(candidates, key=lambda item: (-item["score"], item["code"]))

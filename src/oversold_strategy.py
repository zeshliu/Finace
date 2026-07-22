"""超跌反弹初期筛选和 100 分评分。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_indicators, macd_negative_bars_shrinking, maximum_drawdown
from .scoring import apply_risk_deductions, clamp
from .strategy_utils import build_chart_data, finite_or_none, passes_basic_filters


def oversold_conditions(history: pd.DataFrame, config: dict) -> dict[str, bool]:
    cfg = config["oversold"]
    if history is None or history.empty:
        return {}
    recent5 = history.tail(5)
    latest = history.iloc[-1]
    return {
        "20日最大回撤超过12%": maximum_drawdown(history["close"], 20) <= float(cfg["max_drawdown_20"]),
        "10日跌幅超过8%": float(latest.get("return_10", 0) or 0) <= float(cfg["return_10"]),
        "KDJ最近5日处于低位": bool(((recent5["k"] < float(cfg["kdj_low"])) | (recent5["d"] < float(cfg["kdj_low"]))).any()),
        "J值最近5日低于0": bool((recent5["j"] < float(cfg["j_low"])).any()),
        "RSI6最近5日低于25": bool((recent5["rsi6"] < float(cfg["rsi6_low"])).any()),
        "最近5日跌破布林下轨": bool((recent5["close"] < recent5["boll_lower"]).fillna(False).any()),
    }


def _state_and_score(history: pd.DataFrame, config: dict) -> tuple[dict, dict, list[str]]:
    cfg = config["oversold"]
    latest, previous = history.iloc[-1], history.iloc[-2]
    recent5 = history.tail(5)
    reasons: list[str] = []
    components: dict[str, float] = {}

    shrinking = macd_negative_bars_shrinking(history["macd_hist"], 3)
    dif_up = latest["dif"] > previous["dif"]
    low_cross = previous["dif"] <= previous["dea"] and latest["dif"] > latest["dea"] and latest["dif"] < 0
    approaching_cross = latest["dif"] <= latest["dea"] and (latest["dea"] - latest["dif"]) < (previous["dea"] - previous["dif"])
    macd_score = (8 if shrinking else 0) + (5 if dif_up else 0) + (7 if (low_cross or approaching_cross) else 0)
    components["macd"] = min(20, macd_score)
    if shrinking:
        reasons.append("MACD负柱连续缩短")
    if low_cross:
        reasons.append("MACD低位金叉")
    elif approaching_cross:
        reasons.append("MACD接近金叉")
    elif dif_up:
        reasons.append("DIF开始向上")

    kdj_cross = previous["k"] <= previous["d"] and latest["k"] > latest["d"] and min(latest["k"], latest["d"]) < 35
    j_rebound = recent5["j"].min() < float(cfg["kdj_low"]) and latest["j"] > previous["j"]
    kdj_low = bool(((recent5["k"] < float(cfg["kdj_low"])) | (recent5["d"] < float(cfg["kdj_low"]))).any())
    components["kdj"] = min(20, (10 if kdj_cross else 0) + (6 if j_rebound else 0) + (4 if kdj_low else 0))
    if kdj_cross:
        reasons.append("KDJ低位金叉")
    if j_rebound:
        reasons.append("J值从低位回升")

    above_ma5 = latest["close"] >= latest["ma5"]
    crossed_ma5 = previous["close"] < previous["ma5"] and above_ma5
    ma5_rising = latest["ma5"] >= previous["ma5"] * 0.999
    components["ma"] = min(15, (7 if above_ma5 else 0) + (5 if crossed_ma5 else 0) + (3 if ma5_rising else 0))
    if crossed_ma5:
        reasons.append("收盘价重新站上MA5")
    if ma5_rising:
        reasons.append("MA5走平或向上")

    volume_ma5 = float(latest.get("volume_ma5", 0) or 0)
    volume_ratio = float(latest["volume"] / volume_ma5) if volume_ma5 > 0 else 0
    if float(cfg["volume_ratio_min"]) <= volume_ratio <= float(cfg["volume_ratio_max"]):
        components["volume"] = 15
        reasons.append("成交量温和放大")
    elif 0.9 <= volume_ratio < float(cfg["volume_ratio_min"]):
        components["volume"] = 8
    elif float(cfg["volume_ratio_max"]) < volume_ratio <= 3.2:
        components["volume"] = 5
    else:
        components["volume"] = 0

    conditions = oversold_conditions(history, config)
    condition_count = sum(conditions.values())
    drawdown = maximum_drawdown(history["close"], 20)
    components["oversold"] = min(10, condition_count * 2 + (2 if drawdown <= -0.18 else 0))
    reasons.extend([name for name, matched in conditions.items() if matched][:3])

    recent_rsi_min = recent5["rsi6"].min()
    components["rsi"] = 10 if recent_rsi_min < float(cfg["rsi6_low"]) and latest["rsi6"] > recent_rsi_min else (6 if latest["rsi6"] < 35 else 2)

    candle_range = max(float(latest["high"] - latest["low"]), 1e-9)
    upper_shadow = float(latest["high"] - max(latest["open"], latest["close"])) / candle_range
    components["candle"] = 5 if latest["close"] >= latest["open"] and upper_shadow < 0.4 else (2 if upper_shadow < 0.5 else 0)

    avg_amount = float(history["amount"].tail(20).mean()) if "amount" in history else 0
    min_amount = float(config["screening"]["min_avg_amount_20"])
    components["liquidity"] = round(clamp(avg_amount / (min_amount * 3)) * 5, 1)
    states = {
        "macd": "低位金叉" if low_cross else ("负柱缩短" if shrinking else ("DIF向上" if dif_up else "弱势整理")),
        "kdj": "低位金叉" if kdj_cross else ("低位回升" if j_rebound else "等待确认"),
        "ma": "重上MA5" if crossed_ma5 else ("MA5向上" if ma5_rising else "MA5向下"),
        "volume_ratio": volume_ratio,
        "upper_shadow": upper_shadow,
        "conditions": conditions,
    }
    return states, components, list(dict.fromkeys(reasons))


def evaluate_oversold(spot: dict | pd.Series, raw_history: pd.DataFrame, config: dict) -> dict | None:
    history = add_indicators(raw_history)
    passed, _ = passes_basic_filters(spot, history, config)
    if not passed or len(history) < 60:
        return None
    conditions = oversold_conditions(history, config)
    if sum(conditions.values()) < int(config["oversold"]["min_conditions"]):
        return None

    states, components, reasons = _state_and_score(history, config)
    latest = history.iloc[-1]
    base_score = sum(components.values())
    deductions = []
    if states["upper_shadow"] > 0.5:
        deductions.append(("上影线较长，反弹抛压仍需观察", 5))
    if latest["close"] < latest["ma20"]:
        deductions.append(("仍在MA20下方，中期趋势尚未扭转", 2))
    score, risks = apply_risk_deductions(base_score, deductions)
    if not risks:
        risks = ["技术信号可能失效，请结合市场环境并控制风险"]

    previous_close = float(history["close"].iloc[-2])
    change_pct = (float(latest["close"]) / previous_close - 1) * 100 if previous_close else 0
    code = str(spot.get("code", latest.get("code", ""))).zfill(6)
    return {
        "code": code,
        "name": str(spot.get("name", code)),
        "price": finite_or_none(latest["close"], 2),
        "change_pct": finite_or_none(change_pct, 2),
        "score": score,
        "score_components": {key: round(value, 1) for key, value in components.items()},
        "macd_state": states["macd"],
        "kdj_state": states["kdj"],
        "ma_state": states["ma"],
        "volume_ratio_5": finite_or_none(states["volume_ratio"], 2),
        "return_20_pct": finite_or_none(float(latest.get("return_20", 0)) * 100, 2),
        "rsi6": finite_or_none(latest["rsi6"], 1),
        "reasons": reasons,
        "risks": risks,
        "detail": {"chart": build_chart_data(history, int(config["data"]["chart_days"]))},
    }


def scan_oversold(spot_frame: pd.DataFrame, histories: dict[str, pd.DataFrame], config: dict) -> list[dict]:
    candidates = []
    for _, spot in spot_frame.iterrows():
        code = str(spot["code"]).zfill(6)
        history = histories.get(code)
        if history is None or history.empty:
            continue
        try:
            item = evaluate_oversold(spot, history, config)
            if item and item["score"] >= float(config["oversold"]["score_threshold"]):
                candidates.append(item)
        except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError):
            continue
    return sorted(candidates, key=lambda item: (-item["score"], item["code"]))


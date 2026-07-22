"""隔夜高开统计、尾盘质量判断与评分。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .scoring import apply_risk_deductions, clamp, scaled
from .strategy_utils import build_chart_data, finite_or_none, passes_basic_filters


def calculate_gap_statistics(history: pd.DataFrame, days: int = 60, recent_count: int = 20) -> dict:
    """次日开盘收益 = 次日开盘 / 当日收盘 - 1。"""
    if history is None or history.empty or not {"date", "open", "close"}.issubset(history.columns):
        return {"sample_count": 0, "recent_results": []}
    ordered = history.sort_values("date").copy()
    ordered["next_date"] = ordered["date"].shift(-1)
    ordered["next_open_return"] = pd.to_numeric(ordered["open"], errors="coerce").shift(-1) / pd.to_numeric(ordered["close"], errors="coerce").replace(0, np.nan) - 1
    sample = ordered.dropna(subset=["next_open_return"]).tail(days)
    if sample.empty:
        return {"sample_count": 0, "recent_results": []}
    gaps = sample["next_open_return"]
    recent = []
    for _, row in sample.tail(recent_count).iterrows():
        recent.append(
            {
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "next_date": pd.to_datetime(row["next_date"]).strftime("%Y-%m-%d"),
                "return_pct": finite_or_none(row["next_open_return"] * 100, 2),
                "high_open": bool(row["next_open_return"] > 0),
            }
        )
    return {
        "sample_count": int(len(sample)),
        "high_open_rate": float((gaps > 0).mean()),
        "average_open_return": float(gaps.mean()),
        "median_open_return": float(gaps.median()),
        "above_0_5_probability": float((gaps > 0.005).mean()),
        "above_1_probability": float((gaps > 0.01).mean()),
        "below_minus_1_probability": float((gaps < -0.01).mean()),
        "max_low_open": float(gaps.min()),
        "recent_high_open_rate": float((gaps.tail(min(20, len(gaps))) > 0).mean()),
        "recent_results": recent,
    }


def analyze_tail(spot: dict | pd.Series, history: pd.DataFrame, minute: pd.DataFrame) -> dict | None:
    if minute is None or len(minute) < 6:
        return None
    minute = minute.sort_values("datetime")
    prices = pd.to_numeric(minute["close"], errors="coerce").dropna()
    if len(prices) < 6:
        return None
    current = float(prices.iloc[-1])
    high = float(pd.to_numeric(minute["high"], errors="coerce").max())
    low = float(pd.to_numeric(minute["low"], errors="coerce").min())
    day_open = float(pd.to_numeric(minute["open"], errors="coerce").dropna().iloc[0])
    last_30_start = prices.iloc[-7] if len(prices) >= 7 else prices.iloc[0]
    last_30_return = current / last_30_start - 1 if last_30_start else 0
    position = (current - low) / (high - low) if high > low else 0.5

    volumes = pd.to_numeric(minute.get("volume", 0), errors="coerce").fillna(0)
    amounts = pd.to_numeric(minute.get("amount", 0), errors="coerce").fillna(0)
    if "average" in minute and pd.notna(minute["average"].iloc[-1]):
        vwap = float(minute["average"].iloc[-1])
    else:
        vwap = float(amounts.sum() / (volumes.sum() * 100)) if volumes.sum() > 0 else current
    if not np.isfinite(vwap) or vwap <= 0:
        vwap = current

    average_volume_20 = float(pd.to_numeric(history["volume"], errors="coerce").tail(20).mean())
    current_volume = float(spot.get("volume", volumes.sum()) or volumes.sum())
    volume_ratio = current_volume / average_volume_20 if average_volume_20 > 0 else 0
    candle_range = max(high - low, 1e-9)
    upper_shadow_ratio = (high - max(day_open, current)) / candle_range
    recent_prices = prices.tail(4).to_numpy()
    strengthening = bool(len(recent_prices) >= 3 and recent_prices[-1] > recent_prices[0] and np.polyfit(np.arange(len(recent_prices)), recent_prices, 1)[0] > 0)
    return {
        "current": current,
        "day_open": day_open,
        "high": high,
        "low": low,
        "last_30_return": float(last_30_return),
        "range_position": float(position),
        "vwap": vwap,
        "above_vwap": current >= vwap * 0.998,
        "volume_ratio": float(volume_ratio),
        "upper_shadow_ratio": float(upper_shadow_ratio),
        "strengthening": strengthening,
    }


def evaluate_overnight(
    spot: dict | pd.Series,
    raw_history: pd.DataFrame,
    minute: pd.DataFrame,
    config: dict,
    generated_at: str,
) -> dict | None:
    history = add_indicators(raw_history)
    passed, _ = passes_basic_filters(spot, history, config)
    if not passed:
        return None
    cfg = config["overnight"]
    stats = calculate_gap_statistics(history, int(config["data"]["overnight_stat_days"]), int(config["data"]["recent_gap_results"]))
    if stats.get("sample_count", 0) < int(cfg["min_samples"]):
        return None
    if stats["below_minus_1_probability"] > float(cfg["max_low_open_probability"]):
        return None

    tail = analyze_tail(spot, history, minute)
    if not tail:
        return None
    previous_close = float(pd.to_numeric(history["close"], errors="coerce").iloc[-1])
    if pd.to_datetime(history["date"].iloc[-1]).date() == pd.to_datetime(minute["datetime"].iloc[-1]).date() and len(history) >= 2:
        previous_close = float(history["close"].iloc[-2])
    day_change = tail["current"] / previous_close - 1 if previous_close else 0
    if not (float(cfg["change_min"]) <= day_change <= float(cfg["change_max"])):
        return None
    if tail["last_30_return"] <= float(cfg["tail_drop_limit"]):
        return None
    if tail["range_position"] < float(cfg["min_range_position"]):
        return None
    if tail["upper_shadow_ratio"] > float(cfg["long_upper_shadow_ratio"]) and tail["volume_ratio"] > float(cfg["abnormal_volume_ratio"]):
        return None

    components = {
        "history_high_open": scaled(stats["high_open_rate"], 0.40, 0.70, 20),
        "history_average": scaled(stats["average_open_return"], -0.003, 0.012, 15),
        "recent_stability": scaled(stats["recent_high_open_rate"], 0.35, 0.70, 10),
        "low_open_risk": round((1 - clamp(stats["below_minus_1_probability"] / float(cfg["max_low_open_probability"]))) * 10, 2),
        "sample_stability": round(clamp(stats["sample_count"] / int(config["data"]["overnight_stat_days"])) * 5, 2),
        "tail_trend": scaled(tail["last_30_return"], -0.005, 0.015, 10),
        "tail_volume_price": round((5 if 0.45 <= tail["volume_ratio"] <= 1.8 else 2) + (5 if tail["strengthening"] else 1), 2),
        "close_position": scaled(tail["range_position"], 0.50, 0.90, 8),
        "above_vwap": 5 if tail["above_vwap"] else 0,
        "daily_candle": 4 if tail["current"] >= tail["day_open"] and tail["upper_shadow_ratio"] < 0.35 else 2,
        "liquidity": round(clamp(float(history["amount"].tail(20).mean()) / (float(config["screening"]["min_avg_amount_20"]) * 3)) * 3, 2),
    }
    base_score = sum(components.values())
    deductions = []
    if tail["last_30_return"] < 0:
        deductions.append(("尾盘30分钟仍小幅回落", 4))
    if not tail["above_vwap"]:
        deductions.append(("当前价略低于当日均价", 3))
    if tail["upper_shadow_ratio"] > 0.35:
        deductions.append(("日内上影偏长", 4))
    score, risks = apply_risk_deductions(base_score, deductions)
    if not risks:
        risks = ["隔夜统计不代表未来表现，次日可能受消息与市场波动影响"]

    code = str(spot.get("code", "")).zfill(6)
    return {
        "code": code,
        "name": str(spot.get("name", code)),
        "price": finite_or_none(tail["current"], 2),
        "change_pct": finite_or_none(day_change * 100, 2),
        "high_open_rate_pct": finite_or_none(stats["high_open_rate"] * 100, 1),
        "average_open_return_pct": finite_or_none(stats["average_open_return"] * 100, 2),
        "median_open_return_pct": finite_or_none(stats["median_open_return"] * 100, 2),
        "above_0_5_probability_pct": finite_or_none(stats["above_0_5_probability"] * 100, 1),
        "above_1_probability_pct": finite_or_none(stats["above_1_probability"] * 100, 1),
        "below_minus_1_probability_pct": finite_or_none(stats["below_minus_1_probability"] * 100, 1),
        "max_low_open_pct": finite_or_none(stats["max_low_open"] * 100, 2),
        "last_30_change_pct": finite_or_none(tail["last_30_return"] * 100, 2),
        "range_position_pct": finite_or_none(tail["range_position"] * 100, 1),
        "tail_score": round(sum(components[key] for key in ("tail_trend", "tail_volume_price", "close_position", "above_vwap", "daily_candle")), 1),
        "score": score,
        "score_components": {key: round(value, 1) for key, value in components.items()},
        "risks": risks,
        "updated_at": generated_at,
        "detail": {
            "chart": build_chart_data(history, int(config["data"]["chart_days"])),
            "gap_stats": stats,
        },
    }


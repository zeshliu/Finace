"""评分通用工具。"""

from __future__ import annotations

from collections.abc import Iterable


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def scaled(value: float, low: float, high: float, points: float) -> float:
    if high == low:
        return float(points if value >= high else 0)
    return round(clamp((value - low) / (high - low)) * points, 2)


def apply_risk_deductions(base_score: float, deductions: Iterable[tuple[str, float]]) -> tuple[float, list[str]]:
    """应用风险扣分并返回最终分数与实际触发的风险说明。"""
    score = float(base_score)
    risks: list[str] = []
    for reason, points in deductions:
        if points > 0:
            score -= float(points)
            risks.append(reason)
    return round(clamp(score, 0, 100), 1), risks


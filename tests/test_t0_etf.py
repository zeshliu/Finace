from __future__ import annotations

import numpy as np
import pandas as pd

from src.t0_etf_strategy import classify_t0_etf, score_t0_etf


def make_etf_history(rows: int = 90, amount: float = 120_000_000) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=rows)
    trend = np.linspace(8.0, 10.5, rows)
    wave = np.sin(np.linspace(0, 10, rows)) * 0.08
    close = trend + wave
    return pd.DataFrame(
        {
            "date": dates,
            "code": "518880",
            "open": close * 0.997,
            "high": close * 1.012,
            "low": close * 0.988,
            "close": close,
            "volume": np.linspace(8_000_000, 13_000_000, rows),
            "amount": np.full(rows, amount),
        }
    )


def test_t0_etf_classification_includes_allowed_assets_and_excludes_stock_etfs():
    assert classify_t0_etf("黄金ETF华安", "指数型-其他", "518880") == "黄金ETF"
    assert classify_t0_etf("豆粕ETF华夏", "指数型-其他", "159985") == "商品ETF"
    assert classify_t0_etf("国债ETF国泰", "指数型-固收", "511010") == "债券ETF"
    assert classify_t0_etf("纳指ETF国泰", "指数型-海外股票", "513100") == "跨境ETF"
    assert classify_t0_etf("港股通互联网ETF", "指数型-股票", "520790") == "跨境ETF"
    assert classify_t0_etf("货币ETF易方达", "", "159001") == "货币ETF"
    assert classify_t0_etf("沪深300ETF", "指数型-股票", "510300") is None
    assert classify_t0_etf("黄金股ETF", "指数型-股票", "517520") is None
    assert classify_t0_etf("自由现金流ETF", "指数型-股票", "159201") is None


def test_t0_etf_score_contains_requested_metrics(config):
    spot = {
        "code": "518880",
        "name": "黄金ETF华安",
        "fund_type": "指数型-其他",
        "price": 10.55,
        "pct_change": 1.2,
        "volume": 14_000_000,
        "amount": 150_000_000,
        "volume_ratio": 1.35,
    }
    result = score_t0_etf(spot, make_etf_history(), config)
    assert result is not None
    assert result["category"] == "黄金ETF"
    assert set(result["score_components"]) == {"MACD", "KDJ", "均线", "ATR", "成交量与量比", "RSI"}
    assert 0 <= result["score"] <= 100
    assert result["atr14"] > 0
    assert result["avg_amplitude_20"] >= 0.8
    assert result["volume_ratio"] == 1.35


def test_t0_etf_rejects_insufficient_liquidity(config):
    spot = {
        "code": "518880",
        "name": "黄金ETF华安",
        "fund_type": "指数型-其他",
        "price": 10.5,
        "pct_change": 0.2,
        "volume": 100,
        "amount": 10_000,
        "volume_ratio": 0.1,
    }
    assert score_t0_etf(spot, make_etf_history(amount=1_000_000), config) is None

from __future__ import annotations

import numpy as np

from src.indicators import add_indicators
from src.oversold_strategy import oversold_conditions
from src.scoring import apply_risk_deductions
from src.strategy_utils import passes_basic_filters


def test_price_boundaries_are_inclusive(config, history_factory):
    for price in (10.0, 20.0):
        history = history_factory(close=price)
        spot = {"code": "600000", "name": "浦发银行", "price": price, "volume": 1000}
        passed, reasons = passes_basic_filters(spot, history, config)
        assert passed, reasons


def test_prices_outside_boundaries_are_excluded(config, history_factory):
    history = history_factory(close=9.99)
    spot = {"code": "600000", "name": "浦发银行", "price": 9.99, "volume": 1000}
    passed, reasons = passes_basic_filters(spot, history, config)
    assert not passed
    assert "价格不在设定区间" in reasons


def test_st_is_excluded(config, history_factory):
    spot = {"code": "600000", "name": "*ST示例", "price": 15, "volume": 1000}
    passed, reasons = passes_basic_filters(spot, history_factory(), config)
    assert not passed
    assert "ST股票" in reasons


def test_oversold_conditions_detect_decline(config, history_factory):
    history = history_factory()
    history.loc[history.index[-25]:, "close"] = np.linspace(18, 13, 25)
    history.loc[history.index[-25]:, "open"] = history.loc[history.index[-25]:, "close"] * 1.005
    history.loc[history.index[-25]:, "high"] = history.loc[history.index[-25]:, "close"] * 1.01
    history.loc[history.index[-25]:, "low"] = history.loc[history.index[-25]:, "close"] * 0.98
    enriched = add_indicators(history)
    conditions = oversold_conditions(enriched, config)
    assert conditions["20日最大回撤超过12%"]
    assert conditions["10日跌幅超过8%"]
    assert sum(conditions.values()) >= 2


def test_overnight_scope_and_relaxed_thresholds(config):
    assert config["data"]["intraday_shortlist"] == 300
    assert config["overnight"]["score_threshold"] == 65
    assert config["overnight"]["basic_return_filter_enabled"] is False
    assert config["overnight"]["change_filter_enabled"] is False
    assert config["overnight"]["tail_drop_filter_enabled"] is False
    assert config["overnight"]["range_position_filter_enabled"] is False


def test_overnight_basic_filter_allows_large_price_moves(config, history_factory):
    history = history_factory(close=12.0)
    history.loc[history.index[-1], ["open", "high", "low", "close"]] = [14.5, 15.1, 14.4, 15.0]
    spot = {"code": "600000", "name": "示例股票", "price": 15.0, "volume": 1_000_000}

    regular_passed, regular_reasons = passes_basic_filters(spot, history, config)
    overnight_passed, overnight_reasons = passes_basic_filters(
        spot,
        history,
        config,
        apply_return_limits=False,
    )

    assert not regular_passed
    assert "当天涨停或接近涨停" in regular_reasons
    assert "近3日累计涨幅过大" in regular_reasons
    assert overnight_passed, overnight_reasons


def test_risk_deduction_reduces_score_and_floors_at_zero():
    score, risks = apply_risk_deductions(80, [("长上影", 5), ("跳水", 10)])
    assert score == 65
    assert risks == ["长上影", "跳水"]
    assert apply_risk_deductions(5, [("严重风险", 20)])[0] == 0

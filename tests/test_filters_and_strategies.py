from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import add_indicators
from src.oversold_strategy import oversold_conditions
from src.scoring import apply_risk_deductions
from src.pipeline import preliminary_spot_filter
from src.strategy_utils import passes_basic_filters


def test_price_boundaries_are_inclusive(config, history_factory):
    for price in (0.0, 40.0):
        history = history_factory(close=price)
        spot = {"code": "600000", "name": "浦发银行", "price": price, "volume": 1000}
        passed, reasons = passes_basic_filters(spot, history, config)
        assert passed, reasons


def test_prices_outside_boundaries_are_excluded(config, history_factory):
    history = history_factory(close=40.01)
    spot = {"code": "600000", "name": "浦发银行", "price": 40.01, "volume": 1000}
    passed, reasons = passes_basic_filters(spot, history, config)
    assert not passed
    assert "价格不在设定区间" in reasons


def test_st_is_excluded(config, history_factory):
    spot = {"code": "600000", "name": "*ST示例", "price": 15, "volume": 1000}
    passed, reasons = passes_basic_filters(spot, history_factory(), config)
    assert not passed
    assert "ST股票" in reasons


def test_only_shanghai_and_shenzhen_main_boards_are_allowed(config, history_factory):
    allowed_codes = ("600000", "601398", "603019", "605001", "000001", "001696", "002594", "003816")
    excluded_codes = ("688001", "689009", "300750", "301001", "920000")
    history = history_factory()

    for code in allowed_codes:
        passed, reasons = passes_basic_filters(
            {"code": code, "name": "主板股票", "price": 15, "volume": 1000},
            history,
            config,
        )
        assert passed, (code, reasons)

    for code in excluded_codes:
        passed, reasons = passes_basic_filters(
            {"code": code, "name": "非主板股票", "price": 15, "volume": 1000},
            history,
            config,
        )
        assert not passed
        assert "不属于沪市主板或深市主板" in reasons


def test_preliminary_spot_filter_removes_non_main_boards(config):
    spot = pd.DataFrame(
        {
            "code": ["600000", "000001", "002594", "688001", "300750", "920000"],
            "name": ["沪主板", "深主板", "原中小板", "科创板", "创业板", "北交所"],
            "price": [15.0] * 6,
            "volume": [1000] * 6,
        }
    )
    result = preliminary_spot_filter(spot, config)
    assert result["code"].tolist() == ["600000", "000001", "002594"]


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


def test_risk_deduction_reduces_score_and_floors_at_zero():
    score, risks = apply_risk_deductions(80, [("长上影", 5), ("跳水", 10)])
    assert score == 65
    assert risks == ["长上影", "跳水"]
    assert apply_risk_deductions(5, [("严重风险", 20)])[0] == 0

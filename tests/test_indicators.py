from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import add_indicators, kdj, macd, macd_negative_bars_shrinking, moving_averages, rsi, safe_divide


def test_macd_calculation_has_expected_start_and_direction():
    result = macd(pd.Series([1, 2, 3, 4, 5], dtype=float))
    assert result.iloc[0].to_dict() == {"dif": 0.0, "dea": 0.0, "macd_hist": 0.0}
    assert result["dif"].iloc[-1] > result["dea"].iloc[-1] > 0


def test_kdj_constant_prices_are_neutral():
    prices = pd.Series([10.0] * 12)
    result = kdj(prices, prices, prices)
    assert np.allclose(result[["k", "d", "j"]].to_numpy(), 50.0)


def test_rsi_rising_and_flat_series():
    assert rsi(pd.Series(range(20), dtype=float), 6).dropna().iloc[-1] == 100
    assert rsi(pd.Series([10.0] * 20), 6).dropna().iloc[-1] == 50


def test_moving_average():
    result = moving_averages(pd.Series(range(1, 11), dtype=float), windows=(5,))
    assert result["ma5"].iloc[-1] == 8
    assert pd.isna(result["ma5"].iloc[3])


def test_macd_negative_bars_shrink():
    assert macd_negative_bars_shrinking(pd.Series([-0.5, -0.3, -0.1])) is True
    assert macd_negative_bars_shrinking(pd.Series([-0.5, -0.6, -0.1])) is False
    assert macd_negative_bars_shrinking(pd.Series([-0.5, 0.1, 0.2])) is False


def test_empty_data_and_divide_by_zero():
    assert add_indicators(pd.DataFrame()).empty
    assert safe_divide(4, 0, default=0) == 0
    values = safe_divide(pd.Series([1.0, 2.0]), pd.Series([0.0, 2.0]), default=0)
    assert values.tolist() == [0.0, 1.0]


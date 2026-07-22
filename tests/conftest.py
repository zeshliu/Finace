from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline import load_config


@pytest.fixture
def config():
    return load_config()


def make_history(close=15.0, rows=140, amount=80_000_000):
    dates = pd.bdate_range("2025-01-01", periods=rows)
    closes = np.full(rows, float(close))
    return pd.DataFrame(
        {
            "date": dates,
            "code": "600000",
            "open": closes * 0.995,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(rows, 1_000_000.0),
            "amount": np.full(rows, float(amount)),
        }
    )


@pytest.fixture
def history_factory():
    return make_history


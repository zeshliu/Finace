"""纯 pandas/numpy 技术指标计算，不依赖网络。"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_OHLCV = {"open", "high", "low", "close", "volume"}


def safe_divide(numerator, denominator, default=np.nan):
    """安全除法；分母为 0 时返回 default。"""
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
        result = numerator / denominator
        return result.replace([np.inf, -np.inf], np.nan).fillna(default)
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce")
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False, min_periods=1).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({"dif": dif, "dea": dea, "macd_hist": hist}, index=close.index)


def kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9) -> pd.DataFrame:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    lowest = low.rolling(period, min_periods=1).min()
    highest = high.rolling(period, min_periods=1).max()
    spread = highest - lowest
    rsv = ((close - lowest) / spread.replace(0, np.nan) * 100).fillna(50.0)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"k": k, "d": d, "j": j}, index=close.index)


def rsi(close: pd.Series, period: int = 6) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    values = 100 - 100 / (1 + rs)
    values = values.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    values = values.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return values.rename(f"rsi{period}")


def moving_averages(close: pd.Series, windows=(5, 10, 20, 60)) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce")
    return pd.DataFrame(
        {f"ma{window}": close.rolling(window, min_periods=window).mean() for window in windows},
        index=close.index,
    )


def bollinger_bands(close: pd.Series, window: int = 20, std_multiplier: float = 2.0) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce")
    middle = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame(
        {"boll_mid": middle, "boll_upper": middle + std_multiplier * std, "boll_lower": middle - std_multiplier * std},
        index=close.index,
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ATR；首个有效值至少需要 period 个真实波幅样本。"""
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().rename(f"atr{period}")


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """为标准 OHLCV DataFrame 增加本项目所需的全部指标。"""
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    missing = REQUIRED_OHLCV.difference(frame.columns)
    if missing:
        raise ValueError(f"缺少行情列: {', '.join(sorted(missing))}")

    result = frame.copy()
    if "date" in result.columns:
        result = result.sort_values("date")
    else:
        result = result.sort_index()
    for col in REQUIRED_OHLCV.union({"amount"}).intersection(result.columns):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = pd.concat(
        [
            result,
            moving_averages(result["close"]),
            macd(result["close"]),
            kdj(result["high"], result["low"], result["close"]),
            rsi(result["close"], 6),
            bollinger_bands(result["close"]),
            atr(result["high"], result["low"], result["close"], 14),
        ],
        axis=1,
    )
    result["volume_ma5"] = result["volume"].rolling(5, min_periods=5).mean()
    result["amount_ma20"] = result.get("amount", pd.Series(index=result.index, dtype=float)).rolling(20, min_periods=20).mean()
    result["return_10"] = result["close"].pct_change(10, fill_method=None)
    result["return_20"] = result["close"].pct_change(20, fill_method=None)
    result["pct_change_calc"] = result["close"].pct_change(fill_method=None)
    previous_close = result["close"].shift(1)
    result["amplitude_pct"] = safe_divide(result["high"] - result["low"], previous_close) * 100
    result["atr_ma20"] = result["atr14"].rolling(20, min_periods=20).mean()
    return result


def maximum_drawdown(close: pd.Series, window: int = 20) -> float:
    values = pd.to_numeric(close, errors="coerce").dropna().tail(window)
    if values.empty:
        return 0.0
    running_max = values.cummax()
    drawdowns = values / running_max.replace(0, np.nan) - 1
    value = drawdowns.min()
    return float(value) if pd.notna(value) else 0.0


def macd_negative_bars_shrinking(hist: pd.Series, bars: int = 3) -> bool:
    values = pd.to_numeric(hist, errors="coerce").dropna().tail(bars)
    if len(values) < bars or not (values < 0).all():
        return False
    magnitudes = values.abs().to_numpy()
    return bool(np.all(np.diff(magnitudes) < 0))

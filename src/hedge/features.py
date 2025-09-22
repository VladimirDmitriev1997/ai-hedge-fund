"""
Feature engineering utilities for time-series


- All functions are vectorized, and return objects aligned to input index.
- Inputs are pandas Series (or multiple Series for OHLC features).
- Missing values are handled deterministically via `min_periods`; early values are NaN.
"""

from typing import Literal, Tuple

import numpy as np
import pandas as pd

__all__ = [
    # sanity utilities
    "ensure_series",
    "lag",
    # price transforms
    "simple_returns",
    "log_returns",
    # averages / smoothing
    "sma",
    "ema",
    "wma",
    # volatility & dispersion
    "rolling_vol",
    "ewma_vol",
    "zscore",
    # momentum/oscillators
    "rsi",
    "macd",
    # bands & ranges
    "bollinger_bands",
    "true_range",
    "atr",
    # rolling extrema / sums
    "rolling_min",
    "rolling_max",
    "rolling_sum",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def ensure_series(x: pd.Series, name: str | None = None) -> pd.Series:
    """
    Ensure the input is a float Series with a proper name.

    Parameters
    ----------
    x : pd.Series
        Input series-like (assumed already aligned as needed).
    name : str or None
        Optional replacement name.

    Returns
    -------
    pd.Series
        Float64 series aligned to the original index.
    """
    if not isinstance(x, pd.Series):
        raise TypeError("Expected a pandas Series.")
    out = pd.to_numeric(x, errors="coerce")
    if name is not None:
        out = out.rename(name)
    return out.astype(float)


def lag(x: pd.Series, k: int = 1) -> pd.Series:
    """
    Shift the series by k periods into the past.

    Parameters
    ----------
    x : pd.Series
        Input series.
    k : int, default 1
        Positive values shift *down* (past values appear later).

    Returns
    -------
    pd.Series
        Shifted series with NaNs introduced at the start.
    """
    return ensure_series(x).shift(k)


# ---------------------------------------------------------------------
# Price transforms
# ---------------------------------------------------------------------


def simple_returns(close: pd.Series) -> pd.Series:
    """
    Compute simple returns r_t = (C_t / C_{t-1}) - 1.

    Parameters
    ----------
    close : pd.Series
        Close prices.

    Returns
    -------
    pd.Series
        Simple returns, NaN at the first observation.
    """
    c = ensure_series(close, "close")
    return c.pct_change()


def log_returns(close: pd.Series) -> pd.Series:
    """
    Compute log returns ln(C_t) - ln(C_{t-1}).

    Parameters
    ----------
    close : pd.Series
        Close prices.

    Returns
    -------
    pd.Series
        Log returns, NaN at the first observation.
    """
    c = ensure_series(close, "close")
    return np.log(c).diff()


# ---------------------------------------------------------------------
# Averages / smoothing
# ---------------------------------------------------------------------


def sma(x: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average (SMA).

    Parameters
    ----------
    x : pd.Series
        Input series (e.g., close).
    window : int
        Window length (bars). Must be >= 1.

    Returns
    -------
    pd.Series
        SMA with NaNs for the first (window-1) bars.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    return s.rolling(window=window, min_periods=window).mean()


def ema(x: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average (EMA) with span parameter.

    Parameters
    ----------
    x : pd.Series
        Input series.
    span : int
        EMA span (bars). Must be >= 1.

    Returns
    -------
    pd.Series
        EMA with NaNs until enough history accumulates.
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    s = ensure_series(x)
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def wma(x: pd.Series, window: int) -> pd.Series:
    """
    Weighted Moving Average (linearly weighted over the window).

    Parameters
    ----------
    x : pd.Series
        Input series.
    window : int
        Window length (bars). Must be >= 1.

    Returns
    -------
    pd.Series
        WMA with NaNs for the initial (window-1) bars.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    w = np.arange(1, window + 1, dtype=float)
    # use rolling apply with weights (faster than manual loops)
    return s.rolling(window, min_periods=window).apply(
        lambda a: np.dot(a, w) / w.sum(), raw=True
    )


# ---------------------------------------------------------------------
# Volatility & dispersion
# ---------------------------------------------------------------------


def rolling_vol(x: pd.Series, window: int, ddof: int = 0) -> pd.Series:
    """
    Rolling standard deviation (volatility proxy).

    Parameters
    ----------
    x : pd.Series
        Input series (e.g., returns).
    window : int
        Window length.
    ddof : int, default 0
        Delta degrees of freedom (0 for population-like std).

    Returns
    -------
    pd.Series
        Rolling std with NaNs for the warm-up window.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    return s.rolling(window=window, min_periods=window).std(ddof=ddof)


def ewma_vol(x: pd.Series, span: int, ddof: int = 0) -> pd.Series:
    """
    EWMA volatility proxy via exponentially weighted std.

    Parameters
    ----------
    x : pd.Series
        Input series (e.g., returns).
    span : int
        EWMA span (bars). Must be >= 1.
    ddof : int, default 0
        Degrees of freedom for std.

    Returns
    -------
    pd.Series
        EW std with NaNs until enough history accumulates.
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    s = ensure_series(x)
    # pandas doesn't expose ewm.std with min_periods in older versions; use moment trick
    m = s.ewm(span=span, adjust=False, min_periods=span).mean()
    v = (s - m).pow(2).ewm(span=span, adjust=False, min_periods=span).mean()
    out = np.sqrt(v)
    return out


def zscore(x: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score: (x - rolling_mean) / rolling_std.

    Parameters
    ----------
    x : pd.Series
        Input series.
    window : int
        Window length.

    Returns
    -------
    pd.Series
        Z-score with NaNs during warm-up or where std=0.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    m = sma(s, window)
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    out = (s - m) / sd
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------
# Momentum / oscillators
# ---------------------------------------------------------------------


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    window : int, default 14
        Lookback for average gains/losses.

    Returns
    -------
    pd.Series
        RSI in [0, 100] with NaNs during warm-up.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    c = ensure_series(close, "close")
    delta = c.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via EWMs with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val


def macd(
    close: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence/Divergence (MACD).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    fast_span : int, default 12
        Fast EMA span.
    slow_span : int, default 26
        Slow EMA span.
    signal_span : int, default 9
        EMA span for the signal line.

    Returns
    -------
    pd.DataFrame
        Columns:
        - macd : EMA_fast - EMA_slow
        - signal : EMA(macd, signal_span)
        - hist : macd - signal
    """
    if not (fast_span >= 1 and slow_span >= 1 and signal_span >= 1):
        raise ValueError("all spans must be >= 1")
    c = ensure_series(close, "close")
    ema_fast = ema(c, fast_span)
    ema_slow = ema(c, slow_span)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(
        span=signal_span, adjust=False, min_periods=signal_span
    ).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


# ---------------------------------------------------------------------
# Bands & ranges
# ---------------------------------------------------------------------


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands: middle SMA ± num_std * rolling std.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    window : int, default 20
        SMA/std window.
    num_std : float, default 2.0
        Multiplier for the standard deviation.

    Returns
    -------
    pd.DataFrame
        Columns: mid (SMA), upper, lower, bandwidth, percent_b.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    c = ensure_series(close, "close")
    mid = sma(c, window)
    sd = c.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    bandwidth = (upper - lower) / mid
    percent_b = (c - lower) / (upper - lower)
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "bandwidth": bandwidth,
            "percent_b": percent_b,
        }
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    True Range (TR) per bar.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC series (same index).

    Returns
    -------
    pd.Series
        True range: max(high - low, |high - prev_close|, |low - prev_close|).
    """
    h = ensure_series(high, "high")
    l = ensure_series(low, "low")
    c = ensure_series(close, "close")
    prev_close = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """
    Average True Range (ATR) via Wilder's smoothing.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC series.
    window : int, default 14
        Lookback for ATR.

    Returns
    -------
    pd.Series
        ATR series (same units as price).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    tr = true_range(high, low, close)
    # Wilder's smoothing as an EWMA with alpha=1/window
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


# ---------------------------------------------------------------------
# Rolling extrema / sums
# ---------------------------------------------------------------------


def rolling_min(x: pd.Series, window: int) -> pd.Series:
    """
    Rolling minimum.

    Parameters
    ----------
    x : pd.Series
        Input series.
    window : int
        Window length.

    Returns
    -------
    pd.Series
        Min over the window with NaNs during warm-up.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    return s.rolling(window=window, min_periods=window).min()


def rolling_max(x: pd.Series, window: int) -> pd.Series:
    """
    Rolling maximum.

    Parameters
    ----------
    x : pd.Series
        Input series.
    window : int
        Window length.

    Returns
    -------
    pd.Series
        Max over the window with NaNs during warm-up.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    return s.rolling(window=window, min_periods=window).max()


def rolling_sum(x: pd.Series, window: int) -> pd.Series:
    """
    Rolling sum.

    Parameters
    ----------
    x : pd.Series
        Input series.
    window : int
        Window length.

    Returns
    -------
    pd.Series
        Sum over the window with NaNs during warm-up.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    return s.rolling(window=window, min_periods=window).sum()

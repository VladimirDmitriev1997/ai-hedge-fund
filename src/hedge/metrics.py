"""
Feature engineering utilities for time-series.

- All functions are vectorized and aligned to the input index.
- Inputs are pandas Series (or multiple Series for OHLC features).
- Missing values are handled deterministically via `min_periods`; early values are NaN.
- By default, outputs are shifted by one bar (`causal=True`) to enforce causality:
  features at index t use information available up to t-1.
"""

from typing import Literal, Tuple, Optional

import numpy as np
import pandas as pd
from hedge.utils import ensure_series

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
# Internal helpers
# ---------------------------------------------------------------------


def _causal_shift(x: pd.Series, causal: bool, k: int = 1) -> pd.Series:
    """
    Apply a causal shift if requested.

    Parameters
    ----------
    x : pd.Series
        Series to shift.
    causal : bool
        If True, shift by `k` bars.
    k : int, default 1
        Shift magnitude.

    Returns
    -------
    pd.Series
        Shifted or original series.
    """
    return x.shift(k) if causal else x


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


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


def simple_returns(close: pd.Series, *, causal: bool = True) -> pd.Series:
    """
    Compute simple returns r_t = (C_t / C_{t-1}) - 1.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        Simple returns, NaN at the first observation (and shifted if causal).
    """
    c = ensure_series(close, "close")
    r = c.pct_change()
    return _causal_shift(r, causal)


def log_returns(close: pd.Series, *, causal: bool = True) -> pd.Series:
    """
    Compute log returns ln(C_t) - ln(C_{t-1}), with protection against non-positive prices.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        Log returns, NaN at the first observation (and wherever prices are non-positive).
    """
    c = ensure_series(close, "close").astype(float)
    # Guard: non-positive prices → NaN
    c = c.where(c > 0.0, np.nan)
    lr = np.log(c).diff()
    return _causal_shift(lr, causal)


# ---------------------------------------------------------------------
# Averages / smoothing
# ---------------------------------------------------------------------


def sma(x: pd.Series, window: int, *, causal: bool = True) -> pd.Series:
    """
    Simple Moving Average (SMA).

    Parameters
    ----------
    x : pd.Series
        Input series (e.g., close).
    window : int
        Window length (bars). Must be >= 1.
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        SMA with NaNs for the first (window-1) bars (and shifted if causal).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    out = s.rolling(window=window, min_periods=window).mean()
    return _causal_shift(out, causal)


def ema(x: pd.Series, span: int, *, causal: bool = True) -> pd.Series:
    """
    Exponential Moving Average (EMA) with span parameter.

    Parameters
    ----------
    x : pd.Series
        Input series.
    span : int
        EMA span (bars). Must be >= 1.
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        EMA with NaNs until enough history accumulates (shifted if causal).
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    s = ensure_series(x)
    out = s.ewm(span=span, adjust=False, min_periods=span).mean()
    return _causal_shift(out, causal)


def wma(x: pd.Series, window: int, *, causal: bool = True) -> pd.Series:
    """
    Weighted Moving Average (linearly weighted over the window).

    Parameters
    ----------
    x : pd.Series
        Input series.
    window : int
        Window length (bars). Must be >= 1.
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        WMA with NaNs for the initial (window-1) bars (shifted if causal).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    w = np.arange(1, window + 1, dtype=float)
    out = s.rolling(window, min_periods=window).apply(
        lambda a: np.dot(a, w) / w.sum(), raw=True
    )
    return _causal_shift(out, causal)


# ---------------------------------------------------------------------
# Volatility & dispersion
# ---------------------------------------------------------------------


def rolling_vol(
    x: pd.Series, window: int, ddof: int = 0, *, causal: bool = True
) -> pd.Series:
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
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        Rolling std with NaNs for the warm-up window (shifted if causal).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    out = s.rolling(window=window, min_periods=window).std(ddof=ddof)
    return _causal_shift(out, causal)


def ewma_vol(
    x: pd.Series, span: int, ddof: int = 0, *, causal: bool = True
) -> pd.Series:
    """
    EWMA volatility proxy via exponentially weighted variance.

    Parameters
    ----------
    x : pd.Series
        Input series (e.g., returns).
    span : int
        EWMA span (bars). Must be >= 1.
    ddof : int, default 0
        Degrees of freedom for std (kept for API symmetry).
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        EW std with NaNs until enough history accumulates (shifted if causal).

    Notes
    -----
    Uses the moment trick because older pandas may not support `ewm.std` with min_periods.
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    s = ensure_series(x)
    m = s.ewm(span=span, adjust=False, min_periods=span).mean()
    v = (s - m).pow(2).ewm(span=span, adjust=False, min_periods=span).mean()
    out = np.sqrt(v)
    return _causal_shift(out, causal)


def zscore(x: pd.Series, window: int, *, causal: bool = True) -> pd.Series:
    """
    Rolling z-score: (x - rolling_mean) / rolling_std.

    Parameters
    ----------
    x : pd.Series
        Input series.
    window : int
        Window length.
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        Z-score with NaNs during warm-up or where std=0 (shifted if causal).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    m = sma(s, window, causal=False)  # avoid double shift
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    out = (s - m) / sd
    out = out.replace([np.inf, -np.inf], np.nan)
    return _causal_shift(out, causal)


# ---------------------------------------------------------------------
# Momentum / oscillators
# ---------------------------------------------------------------------


def rsi(close: pd.Series, window: int = 14, *, causal: bool = True) -> pd.Series:
    """
    Relative Strength Index (Wilder's).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    window : int, default 14
        Lookback for average gains/losses.
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        RSI in [0, 100] with NaNs during warm-up (shifted if causal).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    c = ensure_series(close, "close").astype(float)
    delta = c.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via EWMs with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    rsi_val = rsi_val.replace([np.inf, -np.inf], np.nan)
    return _causal_shift(rsi_val, causal)


def macd(
    close: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
    ppo: bool = False,
    *,
    causal: bool = True,
) -> pd.DataFrame:
    """
    Moving Average Convergence/Divergence (MACD) or PPO.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    fast_span : int, default 12
        Fast EMA span (must be < slow_span).
    slow_span : int, default 26
        Slow EMA span.
    signal_span : int, default 9
        EMA span for the signal line.
    ppo : bool, default False
        If True, use PPO core = (EMA_fast - EMA_slow) / EMA_slow (scale-invariant).
    causal : bool, default True
        If True, shift all outputs by 1 to enforce causality.

    Returns
    -------
    pd.DataFrame
        Columns:
        - macd/ppo : core line (MACD or PPO)
        - signal   : EMA(core, signal_span)
        - hist     : core - signal
        All columns are shifted if `causal=True`.

    Notes
    -----
    Validates `fast_span < slow_span`. Protects division by zero in PPO mode.
    """
    if not (fast_span >= 1 and slow_span >= 1 and signal_span >= 1):
        raise ValueError("all spans must be >= 1")
    if not (fast_span < slow_span):
        raise ValueError("fast_span must be < slow_span")

    c = ensure_series(close, "close")
    ema_fast = ema(c, fast_span, causal=False)
    ema_slow = ema(c, slow_span, causal=False)

    diff = ema_fast - ema_slow
    if ppo:
        denom = ema_slow.replace(0.0, np.nan)
        core = diff / denom
        core_name = "ppo"
    else:
        core = diff
        core_name = "macd"

    signal_line = core.ewm(
        span=signal_span, adjust=False, min_periods=signal_span
    ).mean()
    hist = core - signal_line

    out = pd.DataFrame({core_name: core, "signal": signal_line, "hist": hist})
    return out.shift(1) if causal else out


# ---------------------------------------------------------------------
# Bands & ranges
# ---------------------------------------------------------------------


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
    *,
    causal: bool = True,
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
    causal : bool, default True
        If True, shift all outputs by 1 to enforce causality.

    Returns
    -------
    pd.DataFrame
        Columns: mid (SMA), upper, lower, bandwidth, percent_b.
        All columns are shifted if `causal=True`.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    c = ensure_series(close, "close")
    mid = sma(c, window, causal=False)
    sd = c.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    denom = (upper - lower).replace(0.0, np.nan)
    bandwidth = (upper - lower) / mid.replace(0.0, np.nan)
    percent_b = (c - lower) / denom

    out = pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "bandwidth": bandwidth,
            "percent_b": percent_b,
        }
    )
    return out.shift(1) if causal else out


def true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, *, causal: bool = True
) -> pd.Series:
    """
    True Range (TR) per bar.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC series (same index).
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        True range: max(high - low, |high - prev_close|, |low - prev_close|) (shifted if causal).

    Notes
    -----
    Uses `prev_close = close.shift(1)` to form the classic TR definition.
    """
    h = ensure_series(high, "high")
    l = ensure_series(low, "low")
    c = ensure_series(close, "close")
    prev_close = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1
    ).max(axis=1)
    return _causal_shift(tr, causal)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    *,
    causal: bool = True,
) -> pd.Series:
    """
    Average True Range (ATR) via Wilder's smoothing.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC series.
    window : int, default 14
        Lookback for ATR.
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        ATR series (same units as price), shifted if causal.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    tr = true_range(high, low, close, causal=False)
    out = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    return _causal_shift(out, causal)


# ---------------------------------------------------------------------
# Rolling extrema / sums
# ---------------------------------------------------------------------


def rolling_min(x: pd.Series, window: int, *, causal: bool = True) -> pd.Series:
    """
    Rolling minimum.

    Parameters
    ----------
    x : pd.Series
        Input series.
    window : int
        Window length.
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        Min over the window with NaNs during warm-up (shifted if causal).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    out = s.rolling(window=window, min_periods=window).min()
    return _causal_shift(out, causal)


def rolling_max(x: pd.Series, window: int, *, causal: bool = True) -> pd.Series:
    """
    Rolling maximum.

    Parameters
    ----------
    x : pd.Series
        Input series.
    window : int
        Window length.
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        Max over the window with NaNs during warm-up (shifted if causal).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    out = s.rolling(window=window, min_periods=window).max()
    return _causal_shift(out, causal)


def rolling_sum(x: pd.Series, window: int, *, causal: bool = True) -> pd.Series:
    """
    Rolling sum.

    Parameters
    ----------
    x : pd.Series
        Input series.
    window : int
        Window length.
    causal : bool, default True
        If True, shift the result by 1 to enforce causality.

    Returns
    -------
    pd.Series
        Sum over the window with NaNs during warm-up (shifted if causal).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = ensure_series(x)
    out = s.rolling(window=window, min_periods=window).sum()
    return _causal_shift(out, causal)

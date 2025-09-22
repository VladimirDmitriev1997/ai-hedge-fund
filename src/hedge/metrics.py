# metrics.py
"""
Performance metrics for backtests and strategies.

Design
------
- Vectorized
- Minimal assumptions about brokerage/accounting.
- Uses the index timestamps when computing time–based quantities (CAGR, etc.).

Conventions
-----------
- `returns` are per-bar fractional returns (e.g., 0.002 = +0.2%). If you model
  fees/slippage, pass *net* returns.
- `equity` is the compounded account value series, typically produced by
  `equity_curve(returns, init_equity=...)`.
- Time scaling is made explicit:
  * `elapsed_from_index(index, unit=...)` converts the span of a series into the
    requested unit (years, months, weeks, days, hours, etc.).
  * Where variance needs a scale factor, we pass `bars_per_unit` (e.g., for 1-hour
    bars with unit="years", `bars_per_unit = 8760`).
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hedge.utils import ensure_series, elapsed_from_index

# Public API (exported names)
__all__ = [
    # equity & pnl
    "equity_curve",
    "pnl_from_equity",
    "pnl_from_returns",
    # returns-based & annualized metrics
    "roi",
    "roi_excess",
    "cagr",
    "excess_cagr",
    "volatility",
    "sharpe",
    # path-dependent risk
    "max_drawdown",
    # calmar variants & wrapper
    "calmar_nominal",
    "calmar_excess",
    "calmar_real_from_cpi",
    "calmar",
    # other descriptive stats
    "hit_rate",
    "turnover",
]


# ---------------------------------------------------------------------
# Equity & PnL
# ---------------------------------------------------------------------


def equity_curve(returns: pd.Series, init_equity: float = 1.0) -> pd.Series:
    """
    Compound per-bar returns into an equity curve.

    Parameters
    ----------
    returns : pd.Series
        Per-bar fractional returns (already net, if you modeled costs).
    init_equity : float, default 1.0
        Starting notional (account currency units).

    Returns
    -------
    pd.Series
        Compounded equity series named 'equity'. Same index as `returns`.
    """
    r = ensure_series(returns, "returns").fillna(0.0)
    e = (1.0 + r).cumprod() * float(init_equity)
    return e.rename("equity")


def pnl_from_equity(equity: pd.Series) -> pd.Series:
    """
    Per-bar PnL as the first difference of the equity curve.

    Parameters
    ----------
    equity : pd.Series
        Equity series in currency units.

    Returns
    -------
    pd.Series
        Per-bar PnL in currency units, aligned to `equity`, name 'pnl'.
    """
    e = ensure_series(equity, "equity")
    return e.diff().fillna(0.0).rename("pnl")


def pnl_from_returns(
    returns: pd.Series,
    init_equity: float = 1.0,
    reinvest: bool = True,
    fixed_notional: Optional[float] = None,
) -> pd.Series:
    """
    Compute per-bar PnL directly from returns.

    Modes
    -----
    reinvest=True (default)
        Compounding mode. Traded notional each bar equals prior equity:
        `PnL_t = E_{t-1} * r_t`.
    reinvest=False and `fixed_notional` provided
        Non-compounding mode: `PnL_t = fixed_notional * r_t`.

    Parameters
    ----------
    returns : pd.Series
        Per-bar fractional returns.
    init_equity : float, default 1.0
        Initial equity used in compounding mode.
    reinvest : bool, default True
        Whether to compound the notional.
    fixed_notional : float or None
        Required when `reinvest=False`; constant traded notional per bar.

    Returns
    -------
    pd.Series
        Per-bar PnL in currency units, name 'pnl'.
    """
    r = ensure_series(returns, "returns").fillna(0.0)

    if reinvest:
        e = equity_curve(r, init_equity=float(init_equity))
        pnl = e.diff().fillna(e.iloc[0] - float(init_equity))
        return pnl.rename("pnl")
    else:
        if fixed_notional is None:
            raise ValueError("Provide fixed_notional when reinvest=False.")
        return (r * float(fixed_notional)).rename("pnl")


# ---------------------------------------------------------------------
# Nominal & excess return metrics
# ---------------------------------------------------------------------


def roi(equity: pd.Series) -> float:
    """
    Total (nominal) return over the evaluation window.

    Definition
    ----------
    ROI = E_T / E_0 − 1

    Parameters
    ----------
    equity : pd.Series
        Equity curve.

    Returns
    -------
    float
        Total nominal return. NaN if series is too short.
    """
    e = ensure_series(equity, "equity")
    if len(e) < 2:
        return float("nan")
    return float(e.iloc[-1] / e.iloc[0] - 1.0)


def roi_excess(equity: pd.Series, rf: float, time_unit: str) -> float:
    """
    Excess total return over a constant per-`time_unit` risk-free rate.

    Definition
    ----------
    Let `T = elapsed_from_index(e.index, unit=time_unit)`.
    `ROI_excess = (E_T/E_0) / (1 + rf)^T − 1`.

    Parameters
    ----------
    equity : pd.Series
        Equity curve.
    rf : float
        Risk-free rate per `time_unit` (e.g., annual rf if time_unit="years").
    time_unit : str
        Unit used to measure elapsed time (e.g., "years", "months", "days", ...).

    Returns
    -------
    float
        Excess total return versus the rf benchmark. NaN if series too short.
    """
    e = ensure_series(equity, "equity")
    if len(e) < 2:
        return float("nan")
    T = elapsed_from_index(e.index, unit=time_unit)
    if T <= 0:
        return float("nan")
    gross = float(e.iloc[-1] / e.iloc[0])
    bench = (1.0 + float(rf)) ** T
    return float(gross / bench - 1.0)


def cagr(equity: pd.Series, time_unit: str) -> float:
    """
    Compound growth rate per `time_unit`, computed from timestamps.

    Definition
    ----------
    Let `T = elapsed_from_index(e.index, unit=time_unit)`.
    `CAGR = (E_T/E_0)^(1/T) − 1`.

    Parameters
    ----------
    equity : pd.Series
        Equity curve.
    time_unit : str
        Unit used to measure elapsed time (e.g., "years", "months", "days", ...).

    Returns
    -------
    float
        Geometric growth rate per `time_unit`. NaN if series too short.
    """
    e = ensure_series(equity, "equity")
    if len(e) < 2:
        return float("nan")
    T = elapsed_from_index(e.index, unit=time_unit)
    if T <= 0:
        return float("nan")
    return float((e.iloc[-1] / e.iloc[0]) ** (1.0 / T) - 1.0)


def excess_cagr(equity: pd.Series, rf_annual: float, time_unit: str) -> float:
    """
    Geometric excess growth rate per `time_unit` over a constant rf.

    Definition
    ----------
    `g_excess = (1 + g_nominal) / (1 + rf) − 1`, where
    `g_nominal = CAGR(equity, time_unit)`.

    Parameters
    ----------
    equity : pd.Series
        Equity curve.
    rf_annual : float
        Risk-free rate per the same `time_unit` you use for `cagr`.
    time_unit : str
        Unit used to compute the growth rate.

    Returns
    -------
    float
        Excess geometric growth rate. NaN if series too short.
    """
    g_nom = cagr(equity, time_unit=time_unit)
    return float((1.0 + g_nom) / (1.0 + float(rf_annual)) - 1.0)


def volatility(
    returns: pd.Series,
    bars_per_unit: float,
    use_excess: bool = False,
    rf: float = 0.0,
    ddof: int = 0,
) -> float:
    """
    Scaled standard deviation of returns (per chosen unit).

    Parameters
    ----------
    returns : pd.Series
        Per-bar returns.
    bars_per_unit : float
        Number of bars in the chosen unit (e.g., 8760 for 1h bars → years).
    use_excess : bool, default False
        If True, subtract per-bar risk-free (rf / bars_per_unit) before std.
    rf : float, default 0.0
        Risk-free rate per the same unit used for scaling.
    ddof : int, default 0
        Degrees of freedom for std (0 = population style).

    Returns
    -------
    float
        Volatility scaled by sqrt(bars_per_unit).
    """
    r = ensure_series(returns, "returns").dropna()
    if use_excess and rf != 0.0:
        r = r - (float(rf) / float(bars_per_unit))
    return float(r.std(ddof=ddof) * np.sqrt(float(bars_per_unit)))


def sharpe(
    returns: pd.Series,
    bars_per_unit: float,
    rf: float = 0.0,
    ddof: int = 0,
) -> float:
    """
    Sharpe ratio: mean excess return over volatility (per chosen unit).

    Definition
    ----------
    Per-bar excess `x_t = r_t − rf / bars_per_unit`.
    `Sharpe ≈ mean(x) / std(x) * sqrt(bars_per_unit)`.

    Parameters
    ----------
    returns : pd.Series
        Per-bar returns.
    bars_per_unit : float
        Number of bars in the chosen unit (e.g., 8760 for 1h bars → years).
    rf : float, default 0.0
        Risk-free rate per the same unit used for scaling.
    ddof : int, default 0
        Degrees of freedom for std.

    Returns
    -------
    float
        Sharpe ratio. NaN if variance is zero or series too short.
    """
    r = ensure_series(returns, "returns").dropna()
    if len(r) == 0:
        return float("nan")
    x = r - (float(rf) / float(bars_per_unit))
    denom = x.std(ddof=ddof)
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float(x.mean() / denom * np.sqrt(float(bars_per_unit)))


# ---------------------------------------------------------------------
# Path-dependent risk
# ---------------------------------------------------------------------


def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Maximum drawdown (MDD) of the equity curve.

    Parameters
    ----------
    equity : pd.Series
        Equity curve.

    Returns
    -------
    tuple
        `(mdd, peak_ts, trough_ts)` where
        - `mdd` is a non-positive float,
        - `peak_ts` is the timestamp of the prior peak,
        - `trough_ts` is the timestamp of the worst trough.
        Returns `(NaN, NaT, NaT)` if input is empty.
    """
    e = ensure_series(equity, "equity")
    if len(e) == 0:
        return float("nan"), pd.NaT, pd.NaT
    runup = e.cummax()
    dd = e / runup - 1.0
    trough = dd.idxmin()
    if trough is pd.NaT:
        return float(0.0), pd.NaT, pd.NaT
    peak = runup.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


# ---------------------------------------------------------------------
# Calmar variants (nominal / excess / rf-difference)
# ---------------------------------------------------------------------


def calmar_nominal(equity: pd.Series) -> float:
    """
    Nominal Calmar ratio.

    Definition
    ----------
    `Calmar = CAGR(equity) / |MDD(equity)|`.

    Parameters
    ----------
    equity : pd.Series
        Equity curve.

    Returns
    -------
    float
        Nominal Calmar ratio. NaN if MDD is zero or series too short.
    """
    g = cagr(equity)
    mdd, *_ = max_drawdown(equity)
    return float(g / abs(mdd) if mdd != 0 else np.nan)


def calmar_excess(equity: pd.Series, rf: float, time_unit: str) -> float:
    """
    Excess Calmar ratio (geometric excess growth in the numerator).

    Definition
    ----------
    `Calmar_excess = ExcessCAGR(equity, rf, time_unit) / |MDD(equity)|`.

    Parameters
    ----------
    equity : pd.Series
        Equity curve.
    rf : float
        Risk-free rate per `time_unit`.
    time_unit : str
        Unit used to compute the growth rate.

    Returns
    -------
    float
        Excess Calmar ratio. NaN if MDD is zero or series too short.
    """
    g_ex = excess_cagr(equity, rf=float(rf), time_unit=time_unit)
    mdd, *_ = max_drawdown(equity)
    return float(g_ex / abs(mdd) if mdd != 0 else np.nan)


def calmar_rf(equity: pd.Series, rf: float = 0.0, time_unit: str = "years") -> float:
    """
    Calmar ratio with simple rf subtraction in the numerator.

    Definition
    ----------
    `Calmar_rf = (CAGR(equity, time_unit) − rf) / |MDD(equity)|`.

    Parameters
    ----------
    equity : pd.Series
        Equity curve.
    rf : float, default 0.0
        Risk-free rate per `time_unit` to subtract from the CAGR.
    time_unit : str, default "years"
        Unit used to compute CAGR.

    Returns
    -------
    float
        Calmar with rf subtraction. NaN if MDD is zero or series too short.
    """
    e = ensure_series(equity, "equity")
    if len(e) < 2:
        return float("nan")

    r_p = cagr(e, time_unit=time_unit)
    numerator = r_p - float(rf)

    mdd, *_ = max_drawdown(e)
    return float(numerator / abs(mdd) if mdd != 0 else np.nan)


def calmar(
    equity: pd.Series,
    mode: str = "nominal",
    rf: float = 0.0,
    time_unit: str = "year",
) -> float:
    """
    Unified Calmar interface.

    Parameters
    ----------
    equity : pd.Series
        Equity curve.
    mode : {'nominal','excess','rf'}, default 'nominal'
        - 'nominal' : `CAGR/|MDD|` (uses `cagr(equity, ...)` as implemented).
        - 'excess'  : `ExcessCAGR(rf, time_unit)/|MDD|` (geometric excess).
        - 'rf'      : `(CAGR − rf)/|MDD|` (simple difference).
    rf : float, default 0.0
        Risk-free rate per `time_unit` (used in 'excess' and 'rf' modes).
    time_unit : str, default "year"
        Unit used for growth metrics (e.g., "years", "months", "days").

    Returns
    -------
    float
        The chosen Calmar ratio. Raises ValueError on an unknown mode.
    """
    mode = str(mode).lower()
    if mode == "nominal":
        return calmar_nominal(equity, time_unit=time_unit)
    elif mode == "excess":
        return calmar_excess(equity, rf=float(rf), time_unit=time_unit)
    elif mode == "rf":
        return calmar_rf(equity, rf=float(rf), time_unit=time_unit)
    else:
        raise ValueError("mode must be one of {'nominal','excess','rf'}")


# ---------------------------------------------------------------------
# Other descriptive stats
# ---------------------------------------------------------------------


def hit_rate(returns: pd.Series) -> float:
    """
    Fraction of bars with strictly positive return.

    Parameters
    ----------
    returns : pd.Series
        Per-bar returns.

    Returns
    -------
    float
        Share in [0, 1] of bars with `returns > 0`. NaN if no valid data.
    """
    r = ensure_series(returns, "returns").dropna()
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def turnover(positions: pd.Series) -> float:
    """
    Sum of absolute position changes per bar (trading intensity proxy).

    Parameters
    ----------
    positions : pd.Series
        Position series (e.g., -1/0/+1 or continuous sizing). NaNs are treated as 0.

    Returns
    -------
    float
        Sum over time of |pos_t − pos_{t-1}|. To scale to a unit (e.g., yearly),
        convert externally using elapsed time or bar counts.
    """
    p = ensure_series(positions, "position").fillna(0.0)
    return float(p.diff().abs().fillna(0.0).sum())

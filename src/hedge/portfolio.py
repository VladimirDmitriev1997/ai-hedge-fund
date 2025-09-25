"""
Portfolio utilities: alignment of weights/returns, portfolio returns,
and conversion from weights to holdings (units).

Conventions
-----------
- Shapes are time-major.
  * weights: (T, M) — applied over [t, t+1], includes CASH column
  * returns: (T, M?) — per-asset simple returns
      - if returns include CASH → shape (T, M)
      - if returns exclude CASH → shape (T, M-1) and pass `rf`
- Prices for holdings conversion:
  * risky assets use their price at time t (same index as weights),
  * CASH has implicit price == 1 (currency units).

Notes
-----
- This module avoids circular deps with `losses.py`. It exposes:
    * _align_weights_and_returns
    * portfolio_returns
  which `losses.py` imports.
- Holdings conversion is provided in two flavors:
    * NumPy/torch arrays (`weights_to_holdings`)
    * Pandas DataFrame (`weights_to_holdings_df`) with MultiIndex support.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, Callable, Sequence, Mapping, Dict, List

import warnings
import numpy as np
import pandas as pd

# Prefer the canonical backend helpers from torch_utils under hedge.fitting,
# but fall back to hedge.torch_utils for environments that re-export there.
try:
    from hedge.fitting.torch_utils import (
        _is_torch,
        _as_array,
        _zeros_like,
        _sum,
    )
except Exception:  # pragma: no cover
    from hedge.fitting.torch_utils import (  # type: ignore
        _is_torch,
        _as_array,
        _zeros_like,
        _sum,
    )

try:
    import torch
    from torch import Tensor as TorchTensor

    _HAS_TORCH = True
except Exception:  # torch optional
    torch = None  # type: ignore
    TorchTensor = Any  # type: ignore
    _HAS_TORCH = False


# ---------------------------------------------------------------------
# Public constants / exports
# ---------------------------------------------------------------------

CASH_COL: str = "CASH"

__all__ = [
    "CASH_COL",
    "_align_weights_and_returns",
    "portfolio_returns",
    "equity_curve_from_returns",
    "weights_to_holdings",
    "HoldingsResult",
    "weights_to_holdings_df",
]


Array = Union[np.ndarray, "TorchTensor"]
CostFn = Callable[..., Array]
CostItem = Union[CostFn, Tuple[CostFn, Mapping[str, Any]]]
CostConfig = Mapping[str, Mapping[str, Any]]


# ---------------------------------------------------------------------
# Alignment & returns
# ---------------------------------------------------------------------


def _align_weights_and_returns(
    weights: Array,
    returns: Array,
    *,
    has_cash_in_returns: Optional[bool] = None,
    rf: float | Array = 0.0,
    cash_col_idx: Optional[int] = None,
) -> Tuple[Array, Array]:
    """
    Align weights (T,M) and per-asset returns (T,M or T,M-1); inject rf into CASH column if needed.

    Parameters
    ----------
    weights : Array
        Portfolio weights, shape (T, M). Rows sum to 1.0; explicit CASH column present.
    returns : Array
        Per-asset simple returns. If `has_cash_in_returns=True`, shape (T, M);
        otherwise shape (T, M-1) and rf is injected into the CASH column.
    has_cash_in_returns : bool, optional
        Whether `returns` already includes the CASH column. If None, inferred by shape.
    rf : float or Array
        Per-period risk-free (CASH) return. Scalar, (T,) or (T,1).
        Used only when `returns` excludes CASH.
    cash_col_idx : int, optional
        Index of the CASH column in `weights`. Defaults to last column (M-1).

    Returns
    -------
    (Array, Array)
        `(W, R_full)` with matching shapes (T, M).
    """
    W = _as_array(weights)
    R = _as_array(returns)

    if W.ndim != 2 or R.ndim != 2:
        raise ValueError("weights and returns must be 2D: (T, M) and (T, M or M-1).")

    T, M = W.shape
    Tr, Mr = R.shape
    if Tr != T:
        raise ValueError(f"time dimension mismatch: weights {T} vs returns {Tr}")

    cash_idx = M - 1 if cash_col_idx is None else int(cash_col_idx)

    if has_cash_in_returns is None:
        has_cash_in_returns = Mr == M

    if has_cash_in_returns:
        if Mr != M:
            raise ValueError(
                "returns claim to include CASH but columns != weights.shape[1]."
            )
        return W, R

    if Mr != M - 1:
        raise ValueError("returns exclude CASH → expected shape (T, M-1).")

    # Build full returns with rf in CASH column
    R_full = _zeros_like(W)
    risky_cols = [j for j in range(M) if j != cash_idx]
    R_full[:, risky_cols] = R  # type: ignore[index]

    rf_arr = _as_array(rf)
    if _is_torch(rf_arr):
        if rf_arr.ndim == 0:
            rf_arr = rf_arr.repeat(T, 1)  # type: ignore[attr-defined]
        elif rf_arr.ndim == 1:
            rf_arr = rf_arr.view(T, 1)  # type: ignore[attr-defined]
    else:
        rf_arr = np.asarray(rf_arr)
        if rf_arr.ndim == 0:
            rf_arr = np.full((T, 1), float(rf_arr))
        elif rf_arr.ndim == 1:
            rf_arr = rf_arr.reshape(T, 1)

    R_full[:, cash_idx] = rf_arr  # type: ignore[index]
    return W, R_full


def portfolio_returns(
    weights: Array,
    asset_returns: Array,
    *,
    rf: float | Array = 0.0,
    has_cash_in_returns: Optional[bool] = None,
    cash_col_idx: Optional[int] = None,
) -> Array:
    """
    Compute per-period portfolio simple returns r_{p,t} = Σ_j w_{t,j} r_{t,j}.

    Parameters
    ----------
    weights : Array
        Portfolio weights, shape (T, M). Rows sum to 1.0; explicit CASH column present.
    asset_returns : Array
        Per-asset simple returns, shape (T, M) if includes CASH,
        or shape (T, M-1) if excludes CASH (then `rf` is used for CASH).
    rf : float or Array, optional
        Per-period risk-free return for CASH; used only if `asset_returns` excludes CASH.
    has_cash_in_returns : bool, optional
        Whether `asset_returns` already contains CASH. If None, inferred by shape.
    cash_col_idx : int, optional
        Index of the CASH column in `weights`. Defaults to last column.

    Returns
    -------
    Array
        Portfolio simple returns, shape (T,).
    """
    W, R = _align_weights_and_returns(
        weights,
        asset_returns,
        has_cash_in_returns=has_cash_in_returns,
        rf=rf,
        cash_col_idx=cash_col_idx,
    )
    return _sum(W * R, axis=1)


def equity_curve_from_returns(
    port_returns: Array,
    *,
    init_equity: float = 1.0,
) -> Array:
    """
    Compound portfolio returns into an equity curve.

    E_t = E_0 * Π_{s=0..t} (1 + r_{p,s})

    Parameters
    ----------
    port_returns : Array
        Portfolio simple returns, shape (T,).
    init_equity : float, default 1.0
        Initial equity multiplier.

    Returns
    -------
    Array
        Equity series, shape (T,).
    """
    r = _as_array(port_returns)
    if _is_torch(r):
        gross = 1.0 + r
        # torch.cumprod along dim=0
        eq = torch.cumprod(gross, dim=0) * float(init_equity)  # type: ignore[attr-defined]
        return eq
    gross = 1.0 + np.asarray(r)
    return np.cumprod(gross, axis=0) * float(init_equity)


# ---------------------------------------------------------------------
# Weights → holdings (units)
# ---------------------------------------------------------------------


def _returns_from_prices_np(P: np.ndarray) -> np.ndarray:
    """
    Simple returns from prices (NumPy), safe for initial row.

    Parameters
    ----------
    P : ndarray
        Prices, shape (T, K).

    Returns
    -------
    ndarray
        Simple returns, shape (T, K); r[0,:] = 0.
    """
    if P.ndim != 2:
        raise ValueError("prices must be 2D (T, K).")
    if (P <= 0).any():
        raise ValueError("Non-positive prices encountered.")
    r = P[1:, :] / P[:-1, :] - 1.0
    r0 = np.zeros((1, P.shape[1]), dtype=P.dtype)
    return np.vstack([r0, r])


def weights_to_holdings(
    weights: Array,
    prices: Array,
    *,
    cash_col_idx: Optional[int] = None,
    init_equity: float = 1.0,
    # If you already have per-asset returns, pass them to compute equity accurately:
    asset_returns: Optional[Array] = None,
    rf: float | Array = 0.0,
    has_cash_in_returns: Optional[bool] = None,
) -> Tuple[Array, Array]:
    """
    Convert weights to holdings (units) per asset, plus the equity path.

    Mechanics
    ---------
    - Equity path E_t is computed from weights and per-asset returns:
        r_p = Σ_j w_{t,j} r_{t,j}   →   E = cumprod(1 + r_p)
      If `asset_returns` is not provided, we derive simple returns from `prices`
      (risky legs only) and assume CASH per-period return = rf (scalar or (T,)).
    - Holdings:
        For risky j: h_{t,j} = (w_{t,j} * E_t) / price_{t,j}
        For CASH:    h_{t,cash} = w_{t,cash} * E_t   (price ≡ 1)

    Parameters
    ----------
    weights : Array
        (T, M) weights including CASH.
    prices : Array
        Prices for risky legs only. If weights include CASH at index k,
        `prices` must have K = M-1 columns; aligned by time (T, K).
    cash_col_idx : int, optional
        Index of CASH within weights; defaults to last column.
    init_equity : float, default 1.0
        Initial equity multiplier.
    asset_returns : Array, optional
        If provided, used for equity path instead of deriving from prices.
        Shape (T, M) if includes CASH, else (T, M-1) and we inject rf.
    rf : float or Array, default 0.0
        Per-period CASH return; used if `asset_returns` excludes CASH or is None.
    has_cash_in_returns : bool, optional
        Whether `asset_returns` already includes CASH.

    Returns
    -------
    (Array, Array)
        (holdings, equity)
        - holdings : (T, M) units for risky assets; CASH leg in currency units.
        - equity   : (T,) equity path.

    Raises
    ------
    ValueError
        On shape mismatches or non-positive prices.
    """
    W = _as_array(weights)
    P = _as_array(prices)
    if _is_torch(W) or _is_torch(P):
        # Holdings conversion is easiest/cleanest in NumPy; convert tensors.
        W_np = np.asarray(W.detach().cpu().numpy() if _is_torch(W) else W)  # type: ignore
        P_np = np.asarray(P.detach().cpu().numpy() if _is_torch(P) else P)  # type: ignore
        use_torch_out = _is_torch(W)
    else:
        W_np = np.asarray(W)
        P_np = np.asarray(P)
        use_torch_out = False

    if W_np.ndim != 2 or P_np.ndim != 2:
        raise ValueError("weights and prices must be 2D.")
    T, M = W_np.shape
    cash_idx = M - 1 if cash_col_idx is None else int(cash_col_idx)
    K = P_np.shape[1]
    if K != M - 1:
        raise ValueError("prices must have M-1 columns (risky legs only).")
    if P_np.shape[0] != T:
        raise ValueError("prices and weights must have the same T.")

    if (P_np <= 0).any():
        raise ValueError("Non-positive prices encountered.")

    # Compute equity path
    if asset_returns is not None:
        rp = portfolio_returns(
            W_np,
            asset_returns,
            rf=rf,
            has_cash_in_returns=has_cash_in_returns,
            cash_col_idx=cash_idx,
        )
        E = equity_curve_from_returns(rp, init_equity=init_equity)
        E_np = np.asarray(E.detach().cpu().numpy() if _is_torch(E) else E)  # type: ignore
    else:
        # Derive risky returns from prices, and inject rf into CASH
        R_risky = _returns_from_prices_np(P_np)  # (T,K)
        # Assemble full M columns by inserting CASH at cash_idx
        R_full = np.zeros((T, M), dtype=P_np.dtype)
        risky_cols = [j for j in range(M) if j != cash_idx]
        R_full[:, risky_cols] = R_risky
        # Inject rf
        rf_arr = np.asarray(rf)
        if rf_arr.ndim == 0:
            rf_arr = np.full((T,), float(rf_arr))
        elif rf_arr.ndim == 2 and rf_arr.shape[1] == 1:
            rf_arr = rf_arr[:, 0]
        elif rf_arr.ndim != 1:
            raise ValueError("rf must be scalar, (T,), or (T,1).")
        R_full[:, cash_idx] = rf_arr
        rp = np.sum(W_np * R_full, axis=1)
        E_np = np.cumprod(1.0 + rp, axis=0) * float(init_equity)

    # Build holdings
    H = np.zeros_like(W_np, dtype=float)
    risky_cols = [j for j in range(M) if j != cash_idx]
    # Map risky columns → price columns in the same order
    # We assume risky columns in weights correspond one-to-one with price columns order.
    # i.e., weights[:, risky_cols[j]] ↔ prices[:, j]
    for j_risky, j_price in zip(risky_cols, range(len(risky_cols))):
        H[:, j_risky] = (W_np[:, j_risky] * E_np) / P_np[:, j_price]

    # CASH holdings in currency units
    H[:, cash_idx] = W_np[:, cash_idx] * E_np

    if use_torch_out:
        H_t = torch.as_tensor(H, dtype=torch.float32, device=W.device)  # type: ignore[attr-defined]
        E_t = torch.as_tensor(E_np, dtype=torch.float32, device=W.device)  # type: ignore[attr-defined]
        return H_t, E_t
    return H, E_np


# ---------------------------------------------------------------------
# Pandas-friendly front-end
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HoldingsResult:
    """
    Result for DataFrame holdings conversion.

    Attributes
    ----------
    holdings : pd.DataFrame
        Units per asset (risky legs) and currency for CASH; index aligned to input.
    equity : pd.Series
        Equity path used for sizing (same index).
    trades : pd.DataFrame
        Delta holdings (units) per period; first row zeros.
    meta : Dict[str, Any]
        Extra info (cash_col, init_equity, notes).
    """

    holdings: pd.DataFrame
    equity: pd.Series
    trades: pd.DataFrame
    meta: Dict[str, Any]


def _extract_price_matrix_for_weights(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cash_col: str = CASH_COL,
    field: str = "close",
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a prices matrix (T, K) in the order of risky columns in weights.

    Supports:
      - prices with flat columns matching risky asset names (close prices),
      - prices with MultiIndex (symbol, field); we pick (asset, <field>).

    Returns
    -------
    (P, risky_cols)
        P : np.ndarray (T, K)
        risky_cols : ordered risky names aligned to P columns.
    """
    if not isinstance(weights.index, pd.DatetimeIndex):
        raise TypeError("weights must have a DatetimeIndex.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex.")
    if not weights.index.equals(prices.index):
        # Align by index, forward-fill last known price if needed
        prices = prices.reindex(weights.index).ffill()

    risky_cols = [c for c in weights.columns if c != cash_col]

    if isinstance(prices.columns, pd.MultiIndex):
        # Expect level names: ("symbol", "field")
        P_df = []
        for a in risky_cols:
            if (a, field) not in prices.columns:
                raise KeyError(f"Missing price column {(a, field)!r} in prices.")
            P_df.append(prices[(a, field)])
        P = np.column_stack([p.to_numpy(dtype=float) for p in P_df])
    else:
        # Flat columns: expect each risky asset present as a column of prices
        for a in risky_cols:
            if a not in prices.columns:
                raise KeyError(f"Missing price column '{a}' in prices.")
        P = prices[risky_cols].to_numpy(dtype=float)

    if (P <= 0).any():
        raise ValueError("Non-positive prices encountered in prices DataFrame.")
    return P, risky_cols


def weights_to_holdings_df(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cash_col: str = CASH_COL,
    field: str = "close",
    init_equity: float = 1.0,
    asset_returns: Optional[pd.DataFrame] = None,
    rf: Union[float, pd.Series] = 0.0,
    has_cash_in_returns: Optional[bool] = None,
) -> HoldingsResult:
    """
    DataFrame wrapper for weights → holdings conversion.

    Parameters
    ----------
    weights : pd.DataFrame
        Weights including `cash_col`; columns = [asset_1, ..., asset_K, cash_col].
    prices : pd.DataFrame
        Price panel; either flat columns named by asset (close prices),
        or MultiIndex (symbol, field) where we pick (asset, `field`).
        Index must align to `weights` (will be aligned with ffill if needed).
    cash_col : str, default 'CASH'
        Name of the CASH column in `weights`.
    field : str, default 'close'
        Field to select from MultiIndex prices.
    init_equity : float, default 1.0
        Initial equity multiplier.
    asset_returns : pd.DataFrame, optional
        Per-asset simple returns for equity computation (see semantics in array API).
        If None, returns are derived from `prices` and `rf`.
    rf : float or pd.Series, default 0.0
        Per-period CASH return; used if `asset_returns` excludes CASH or is None.
    has_cash_in_returns : bool, optional
        Whether `asset_returns` already includes CASH.

    Returns
    -------
    HoldingsResult
        holdings, equity, trades, meta
    """
    if cash_col not in weights.columns:
        raise KeyError(f"weights must include '{cash_col}' column.")

    # Extract prices matrix in risky order (weights' column order)
    P, risky_cols = _extract_price_matrix_for_weights(
        weights, prices, cash_col=cash_col, field=field
    )

    # Prepare array inputs
    W = weights[risky_cols + [cash_col]].to_numpy(dtype=float)
    cash_idx = len(risky_cols)  # since we ordered risky first, then cash

    if asset_returns is not None:
        R = asset_returns.to_numpy(dtype=float)
        H_arr, E_arr = weights_to_holdings(
            W,
            P,
            cash_col_idx=cash_idx,
            init_equity=float(init_equity),
            asset_returns=R,
            rf=(rf.to_numpy() if isinstance(rf, pd.Series) else rf),
            has_cash_in_returns=has_cash_in_returns,
        )
    else:
        H_arr, E_arr = weights_to_holdings(
            W,
            P,
            cash_col_idx=cash_idx,
            init_equity=float(init_equity),
            asset_returns=None,
            rf=(rf.to_numpy() if isinstance(rf, pd.Series) else rf),
            has_cash_in_returns=None,
        )

    # Build holdings DataFrame with same column order as weights
    cols = risky_cols + [cash_col]
    holdings_df = pd.DataFrame(H_arr, index=weights.index, columns=cols)

    # Trades (Δ holdings), first row zeros
    trades_df = holdings_df.diff().fillna(0.0)

    equity_ser = pd.Series(E_arr, index=weights.index, name="equity")

    meta: Dict[str, Any] = {
        "cash_col": cash_col,
        "risky_assets": risky_cols,
        "init_equity": float(init_equity),
        "notes": "Holdings computed as units; CASH leg in currency units (price ≡ 1).",
    }

    return HoldingsResult(
        holdings=holdings_df,
        equity=equity_ser,
        trades=trades_df,
        meta=meta,
    )

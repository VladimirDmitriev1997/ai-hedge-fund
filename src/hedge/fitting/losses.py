# losses.py
"""
Differentiable (and NumPy) trading losses & utilities for strategy fitting.

Design
------
- Backend-agnostic functions that work with either NumPy arrays or PyTorch tensors.
- Portfolio-friendly: weights are full portfolio allocations (rows sum to 1.0), and
  can include a dedicated CASH column.
- Costs modeled from turnover (|Δw|) with fee/slippage in basis points.
- Objectives include log-wealth, Sharpe proxy, and simple drawdown proxy.
- Pure numeric core for autodiff compatibility.

Conventions
-----------
Shapes (time-major):
- weights: (T, M)  — row t is the allocation over assets (including CASH if used).
- returns: (T, M?) — per-asset simple returns over [t, t+1].
    * If returns includes CASH (same M as weights): use it directly.
    * If returns excludes CASH (M == weights.shape[1] - 1): pass rf to fill CASH.
- rf: float or (T,) or (T, 1) — per-period risk-free return for the CASH leg.
- All inputs must be aligned in time; decisions at t apply to returns over [t, t+1].

Notes
-----
- Functions accept either numpy arrays or torch tensors. If torch is unavailable,
  the torch path is simply not used.
- Costs are rates subtracted from returns (i.e., 1 + r_p - cost).
"""


from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor as TorchTensor

    _HAS_TORCH = True
except Exception:
    torch = None
    TorchTensor = Any
    _HAS_TORCH = False


Array = Union[np.ndarray, TorchTensor]


# ---------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------


def _is_torch(x: Any) -> bool:
    """Return True if x is a torch.Tensor."""
    return _HAS_TORCH and isinstance(x, TorchTensor)


def _as_array(x: Any) -> Array:
    """
    Coerce common inputs into a backend array (numpy or torch).

    Parameters
    ----------
    x : Any
        Supports: numpy arrays, torch tensors, Python scalars, sequences.

    Returns
    -------
    Array
        np.ndarray or torch.Tensor, matching input type when possible.
    """
    if _is_torch(x):
        return x
    if isinstance(x, np.ndarray):
        return x

    return np.asarray(x)


def _zeros_like(x: Array) -> Array:
    return torch.zeros_like(x) if _is_torch(x) else np.zeros_like(x)


def _ones_like(x: Array) -> Array:
    return torch.ones_like(x) if _is_torch(x) else np.ones_like(x)


def _abs(x: Array) -> Array:
    return torch.abs(x) if _is_torch(x) else np.abs(x)


def _sum(x: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    if _is_torch(x):
        return torch.sum(x, dim=axis, keepdim=keepdims)
    return np.sum(x, axis=axis, keepdims=keepdims)


def _mean(x: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    if _is_torch(x):
        return torch.mean(x, dim=axis, keepdim=keepdims)
    return np.mean(x, axis=axis, keepdims=keepdims)


#### Future implementations - add ddof


def _std(
    x: Array, axis: Optional[int] = None, keepdims: bool = False, eps: float = 1e-12
) -> Array:
    if _is_torch(x):
        return torch.sqrt(
            torch.clamp(
                torch.var(x, dim=axis, unbiased=False, keepdim=keepdims), min=eps
            )
        )
    out = np.std(x, axis=axis, ddof=0, keepdims=keepdims)
    return np.maximum(out, eps)


def _log1p(x: Array) -> Array:
    return torch.log1p(x) if _is_torch(x) else np.log1p(x)


def _clip(x: Array, lo: float, hi: float) -> Array:
    if _is_torch(x):
        return torch.clamp(x, min=lo, max=hi)
    return np.clip(x, lo, hi)


def _cumsum(x: Array, axis: int = 0) -> Array:
    return torch.cumsum(x, dim=axis) if _is_torch(x) else np.cumsum(x, axis=axis)


def _cumprod(x: Array, axis: int = 0) -> Array:
    return torch.cumprod(x, dim=axis) if _is_torch(x) else np.cumprod(x, axis=axis)


def _maximum(a: Array, b: Array) -> Array:
    if _is_torch(a) or _is_torch(b):
        a2, b2 = _as_array(a), _as_array(b)
        return torch.maximum(a2, b2)
    return np.maximum(a, b)


def _where(mask: Array, a: Array, b: Array) -> Array:
    if _is_torch(mask) or _is_torch(a) or _is_torch(b):
        return torch.where(_as_array(mask) != 0, _as_array(a), _as_array(b))
    return np.where(mask, a, b)


# ---------------------------------------------------------------------
# Core portfolio computations
# ---------------------------------------------------------------------


def _align_weights_and_returns(
    weights: Array,
    returns: Array,
    *,
    has_cash_in_returns: bool | None = None,
    rf: float | Array = 0.0,
    cash_col_idx: Optional[int] = None,
) -> Tuple[Array, Array]:
    """
    Align shapes between weights (T,M) and returns (T,M or T,M-1) and inject rf if needed.

    Parameters
    ----------
    weights : Array
        Full portfolio weights (rows sum to ~1). Shape (T, M).
    returns : Array
        Asset returns. Either:
          - (T, M) including CASH return in the cash column, or
          - (T, M-1) excluding CASH; `rf` is used for CASH.
    has_cash_in_returns : bool or None
        If None, inferred from shape: if returns.shape[1] == weights.shape[1], assume True.
    rf : float or Array
        Per-period risk-free (CASH) return. Broadcasts to (T,1) if needed.
    cash_col_idx : int or None
        Index of the CASH column in `weights` (and returns if present). If None, defaults to LAST column.

    Returns
    -------
    (weights, returns_full) : Tuple[Array, Array]
        Returns array guaranteed to have same (T,M) shape as weights. If CASH returns were missing,
        rf is injected into the CASH column.
    """
    W = _as_array(weights)
    R = _as_array(returns)

    if W.ndim != 2 or R.ndim != 2:
        raise ValueError(
            "weights and returns must be 2D arrays: (T, M) and (T, M or M-1)."
        )

    T, M = W.shape
    Tr, Mr = R.shape

    if Tr != T:
        raise ValueError(
            f"weights and returns must have the same time dimension: {T} vs {Tr}."
        )

    # Decide where CASH is
    cash_idx = M - 1 if cash_col_idx is None else int(cash_col_idx)

    if has_cash_in_returns is None:
        has_cash_in_returns = Mr == M

    if has_cash_in_returns:
        if Mr != M:
            raise ValueError(
                "returns claims to include CASH but shape mismatch with weights."
            )
        return W, R

    if (not has_cash_in_returns) and (Mr != M - 1):
        raise ValueError("If returns exclude CASH, its shape must be (T, M-1).")

    # Build full returns by inserting rf in the cash column
    R_full = _zeros_like(W)
    # risky columns = all indices except cash_idx
    risky_cols = [j for j in range(M) if j != cash_idx]
    if _is_torch(W):
        R_full[:, risky_cols] = R  # type: ignore[index]
        rf_arr = _as_array(rf)
        rf_arr = (
            rf_arr.view(-1, 1) if _is_torch(rf_arr) and rf_arr.ndim == 1 else rf_arr
        )
        if _is_torch(rf_arr):
            if rf_arr.ndim == 0:
                rf_arr = rf_arr.repeat(T, 1)
            elif rf_arr.shape[0] != T:
                rf_arr = rf_arr.repeat(T, 1) if rf_arr.numel() == 1 else rf_arr
        else:
            rf_arr = np.asarray(rf_arr)
            if rf_arr.ndim == 0:
                rf_arr = np.full((T, 1), float(rf_arr))
            elif rf_arr.ndim == 1:
                rf_arr = rf_arr.reshape(T, 1)
        R_full[:, cash_idx] = rf_arr  # type: ignore[index]
    else:
        R_full[:, risky_cols] = R
        rf_arr = np.asarray(rf)
        if rf_arr.ndim == 0:
            rf_arr = np.full((T, 1), float(rf_arr))
        elif rf_arr.ndim == 1:
            rf_arr = rf_arr.reshape(T, 1)
        R_full[:, cash_idx] = rf_arr

    return W, R_full


def portfolio_returns(
    weights: Array,
    asset_returns: Array,
    *,
    rf: float | Array = 0.0,
    has_cash_in_returns: bool | None = None,
    cash_col_idx: Optional[int] = None,
) -> Array:
    """
    Compute per-period **portfolio simple returns** from weights and per-asset returns.

    Parameters
    ----------
    weights : Array, shape (T, M)
        Full portfolio weights (including CASH column). Rows should sum to 1.
    asset_returns : Array, shape (T, M) or (T, M-1)
        Per-asset returns over [t, t+1]. If CASH returns are not provided
        (shape M-1), pass `rf` to fill the CASH column.
    rf : float or Array, default 0.0
        Per-period CASH return when returns exclude CASH.
    has_cash_in_returns : bool or None, default None
        Force interpretation whether `asset_returns` includes CASH. If None, inferred via shape.
    cash_col_idx : int or None, default None
        Index of CASH in the weights (and returns if present). Default: last column.

    Returns
    -------
    Array, shape (T,)
        Per-period portfolio simple returns r_p,t.
    """
    W, R = _align_weights_and_returns(
        weights,
        asset_returns,
        has_cash_in_returns=has_cash_in_returns,
        rf=rf,
        cash_col_idx=cash_col_idx,
    )
    # r_p,t = sum_j w_{t,j} * r_{t,j}
    rp = _sum(W * R, axis=1)
    return rp  # shape (T,)


def costs_turnover(
    weights: Array,
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    per_asset_multipliers: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
) -> Array:
    """
    Compute **per-period cost rates** from turnover |Δw| with proportional fees/slippage.

    Cost model
    ----------
    cost_t = sum_j |w_{t,j} - w_{t-1,j}| * kappa_j(t),
      where kappa_j(t) = (fee_bps + slippage_bps) / 1e4 * multiplier_j(t).

    Parameters
    ----------
    weights : Array, shape (T, M)
        Full portfolio weights (incl. CASH).
    fee_bps : float, default 0.0
        Proportional fee per trade in basis points (e.g., 5 = 5 bps = 0.05%).
    slippage_bps : float, default 0.0
        Additional slippage in bps.
    per_asset_multipliers : Array or None, default None
        If provided, broadcastable to (T, M). Use to scale costs per asset (e.g., spread, vol).
    initial_weights : Array or None, default None
        If provided, used as w_{-1} for the first period; otherwise assumed 0.

    Returns
    -------
    Array, shape (T,)
        Per-period cost **rates** (same units as returns), to be subtracted from r_p.
    """
    W = _as_array(weights)
    T, M = W.shape

    if initial_weights is None:
        w_prev = (
            _zeros_like(W[:1, :]).repeat(1, 1)
            if _is_torch(W)
            else np.zeros((1, M), dtype=W.dtype)
        )
    else:
        w_prev = _as_array(initial_weights)
        if w_prev.ndim == 1:
            w_prev = w_prev.reshape(1, -1)
        if w_prev.shape[1] != M:
            raise ValueError("initial_weights must have shape (M,) or (1, M).")

    if _is_torch(W):
        dw = torch.vstack((W[0:1, :] - w_prev, W[1:, :] - W[:-1, :]))
    else:
        dw = np.vstack((W[0:1, :] - w_prev, W[1:, :] - W[:-1, :]))

    kappa = (float(fee_bps) + float(slippage_bps)) / 1e4
    if kappa == 0.0 and per_asset_multipliers is None:
        return _zeros_like(W[:, 0])

    mult = 1.0
    if per_asset_multipliers is not None:
        mult = _as_array(per_asset_multipliers)

        if _is_torch(mult):
            if mult.ndim == 1:
                mult = (
                    mult.view(1, -1).repeat(T, 1)
                    if mult.numel() == M
                    else mult.view(T, -1)
                )
        else:
            mult = np.asarray(mult)
            if mult.ndim == 1:
                if mult.size == M:
                    mult = np.tile(mult.reshape(1, M), (T, 1))
                elif mult.size == T:
                    mult = np.tile(mult.reshape(T, 1), (1, M))
                else:
                    raise ValueError("per_asset_multipliers 1D must match M or T.")

    abs_dw = _abs(dw)
    if isinstance(mult, (int, float)):
        per_asset_cost = abs_dw * kappa
    else:
        per_asset_cost = abs_dw * (kappa * _as_array(mult))

    cost_t = _sum(per_asset_cost, axis=1)  # (T,)
    return cost_t


def wealth_curve(
    port_ret: Array,
    costs: Optional[Array] = None,
    *,
    init_equity: float = 1.0,
    clip_min: float = -0.999999,
) -> Array:
    """
    Build an equity curve from per-period portfolio returns and cost rates.

    Wealth recursion
    ----------------
    E_{t+1} = E_t * (1 + r_{p,t} - cost_t)

    Parameters
    ----------
    port_ret : Array, shape (T,)
        Portfolio returns per period.
    costs : Array or None, shape (T,), default None
        Per-period cost rates to subtract from returns. If None, treated as 0.
    init_equity : float, default 1.0
        Initial equity E_0.
    clip_min : float, default -0.999999
        Lower bound on (r - cost) to avoid negative growth factors.

    Returns
    -------
    Array, shape (T,)
        Equity curve aligned to the input.
    """
    r = _as_array(port_ret)
    c = _zeros_like(r) if costs is None else _as_array(costs)
    g = _clip(r - c, clip_min, np.inf)  # growth rate per-period
    gross = 1.0 + g
    eq = _cumprod(gross, axis=0) * float(init_equity)
    return eq


# ---------------------------------------------------------------------
# Losses (minimize)
# ---------------------------------------------------------------------


def neg_log_wealth(
    port_ret: Array,
    costs: Optional[Array] = None,
    *,
    eps: float = 1e-12,
) -> Array:
    """
    Negative log-wealth objective (Kelly-style).

    L = -∑_t log(1 + r_{p,t} - cost_t + eps)

    Parameters
    ----------
    port_ret : Array, shape (T,)
        Portfolio returns per period.
    costs : Array or None, shape (T,), default None
        Per-period cost rates.
    eps : float, default 1e-12
        Numerical jitter for stability inside log(1+⋅).

    Returns
    -------
    Array (scalar)
        Loss scalar in the same backend as inputs.
    """
    r = _as_array(port_ret)
    c = _zeros_like(r) if costs is None else _as_array(costs)
    val = _log1p(r - c + (eps if not _is_torch(r) else _as_array(eps)))
    # negative sum
    return -_sum(val)


def neg_sharpe(
    port_ret: Array,
    costs: Optional[Array] = None,
    *,
    annualizer: float,
    rf_per_period: float | Array = 0.0,
    eps: float = 1e-12,
) -> Array:
    """
    Negative Sharpe-like objective (annualized), using excess returns over rf.

    Sharpe ≈ mean(x) / std(x) * sqrt(annualizer), minimized as -Sharpe.

    Parameters
    ----------
    port_ret : Array, shape (T,)
        Portfolio returns per period.
    costs : Array or None, shape (T,), default None
        Per-period cost rates.
    annualizer : float
        Annualization factor (e.g., 365 for daily, 8760 for hourly).
    rf_per_period : float or Array, default 0.0
        Per-period risk-free return to subtract.
    eps : float, default 1e-12
        Numerical guard for std.

    Returns
    -------
    Array (scalar)
        Loss scalar (negative Sharpe).
    """
    r = _as_array(port_ret)
    c = _zeros_like(r) if costs is None else _as_array(costs)
    rf = _as_array(rf_per_period)

    # broadcast rf to match r
    if _is_torch(r):
        if _is_torch(rf) and rf.ndim == 0:
            rf = rf.repeat(r.shape[0])
        elif not _is_torch(rf):
            rf = _as_array(float(rf)) if np.isscalar(rf) else _as_array(rf)
            if rf.ndim == 0:
                rf = _as_array(float(rf)).repeat(r.shape[0])  # type: ignore
    else:
        rf = np.asarray(rf)
        if rf.ndim == 0:
            rf = np.full_like(r, float(rf))

    x = r - c - rf
    mu = _mean(x)
    sd = _std(x, eps=eps)
    sharpe = mu / sd * (annualizer**0.5)
    return -sharpe


def drawdown_surrogate(
    port_ret: Array,
    costs: Optional[Array] = None,
    *,
    init_equity: float = 1.0,
    tau: float = 10.0,
    eps: float = 1e-12,
) -> Array:
    """
    Smooth drawdown-aware loss (heuristic).

    Idea
    ----
    - Build equity E_t from (r_p - cost).
    - Compute running peak P_t = max_{s<=t} E_s.
    - Penalize average **soft drawdown** via softplus(P_t - E_t) / P_t.

    Loss
    ----
    L = mean( softplus( (P_t - E_t) / (P_t + eps) * tau ) / tau )

    Parameters
    ----------
    port_ret : Array, shape (T,)
        Portfolio returns per period.
    costs : Array or None, shape (T,), default None
        Per-period cost rates.
    init_equity : float, default 1.0
        Starting equity.
    tau : float, default 10.0
        Softness/temperature of the penalty.
    eps : float, default 1e-12
        Numerical guard.

    Returns
    -------
    Array (scalar)
        Loss scalar (larger when drawdowns are severe/long).
    """
    eq = wealth_curve(port_ret, costs, init_equity=init_equity)
    # running peak
    if _is_torch(eq):
        # torch.cummax returns (values, indices)
        peak = torch.cummax(eq, dim=0)[0]  # type: ignore
        rel_gap = (peak - eq) / (peak + _as_array(eps))
        soft = torch.nn.functional.softplus(rel_gap * tau) / tau  # type: ignore
        return _mean(soft)
    else:
        peak = np.maximum.accumulate(eq)
        rel_gap = (peak - eq) / (peak + eps)
        soft = np.log1p(np.exp(rel_gap * tau)) / tau  # softplus
        return _mean(soft)


def regularizers(
    weights: Array,
    *,
    l1_turnover: float = 0.0,
    l1_leverage: float = 0.0,
) -> Array:
    """
    Linear penalties for turnover and leverage (promote stability & realism).

    Penalty
    -------
    R = λ_to * mean_t sum_j |Δw_{t,j}|  +  λ_lev * mean_t sum_j |w_{t,j}|

    Parameters
    ----------
    weights : Array, shape (T, M)
        Full portfolio weights (incl. CASH).
    l1_turnover : float, default 0.0
        Coefficient for average L1 turnover.
    l1_leverage : float, default 0.0
        Coefficient for average L1 leverage (absolute weights).

    Returns
    -------
    Array (scalar)
        Penalty scalar.
    """
    if (l1_turnover == 0.0) and (l1_leverage == 0.0):
        # Return a backend scalar zero
        W = _as_array(weights)
        return _sum(W[:, :1] * 0.0)

    W = _as_array(weights)
    T, _ = W.shape

    if _is_torch(W):
        dw = torch.vstack((W[0:1, :], W[1:, :] - W[:-1, :]))
    else:
        dw = np.vstack((W[0:1, :], W[1:, :] - W[:-1, :]))

    pen_to = float(l1_turnover) * _mean(_sum(_abs(dw), axis=1))
    pen_lev = float(l1_leverage) * _mean(_sum(_abs(W), axis=1))
    return pen_to + pen_lev


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------


@dataclass
class Objective:
    """
    Bundle objective and its settings for training loops.

    Attributes
    ----------
    name : str
        One of {"neg_log_wealth", "neg_sharpe", "drawdown"}.
    annualizer : float
        Required for Sharpe (ignored otherwise).
    rf_per_period : float or Array
        Used by Sharpe (excess); ignored by others unless injected upstream.
    init_equity : float
        Used by drawdown (wealth curve).
    eps : float
        Numerical jitter for stability.
    tau : float
        Softness for drawdown surrogate.
    weight_turnover : float
        λ for L1 turnover penalty.
    weight_leverage : float
        λ for L1 leverage penalty.
    """

    name: str = "neg_log_wealth"
    annualizer: float = 365.0
    rf_per_period: float | Array = 0.0
    init_equity: float = 1.0
    eps: float = 1e-12
    tau: float = 10.0
    weight_turnover: float = 0.0
    weight_leverage: float = 0.0


def evaluate_objective(
    weights: Array,
    asset_returns: Array,
    *,
    rf: float | Array = 0.0,
    has_cash_in_returns: bool | None = None,
    cash_col_idx: Optional[int] = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    per_asset_multipliers: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    spec: Objective = Objective(),
) -> Array:
    """
    Compute an end-to-end objective: weights + returns -> r_p, costs -> loss (+ penalties).

    Parameters
    ----------
    weights : Array, shape (T, M)
        Portfolio weights (rows sum to ~1, incl. CASH).
    asset_returns : Array, shape (T, M) or (T, M-1)
        Per-asset returns. If CASH missing, pass `rf`.
    rf : float or Array, default 0.0
        Per-period CASH return when returns exclude CASH.
    has_cash_in_returns : bool or None, default None
        Force whether returns includes CASH. If None, inferred by shape.
    cash_col_idx : int or None, default None
        Index of CASH in weights (and returns if present). Default: last column.
    fee_bps : float, default 0.0
        Proportional fee per trade in bps.
    slippage_bps : float, default 0.0
        Extra slippage in bps.
    per_asset_multipliers : Array or None, default None
        Scale costs per asset/time.
    initial_weights : Array or None, default None
        Initial portfolio (for first-step turnover). If None, assumes 0.
    spec : ObjectiveSpec, default ObjectiveSpec()
        Choose objective and regularizer weights.

    Returns
    -------
    Array (scalar)
        Objective value suitable for minimization (loss).
    """
    # Portfolio returns (inject rf into CASH if needed)
    rp = portfolio_returns(
        weights,
        asset_returns,
        rf=rf,
        has_cash_in_returns=has_cash_in_returns,
        cash_col_idx=cash_col_idx,
    )
    # Costs from turnover
    cost = costs_turnover(
        weights,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        per_asset_multipliers=per_asset_multipliers,
        initial_weights=initial_weights,
    )

    # Base objective
    name = spec.name.lower()
    if name == "neg_log_wealth":
        loss = neg_log_wealth(rp, cost, eps=spec.eps)
    elif name == "neg_sharpe":
        loss = neg_sharpe(
            rp,
            cost,
            annualizer=spec.annualizer,
            rf_per_period=spec.rf_per_period,
            eps=spec.eps,
        )
    elif name in ("drawdown", "drawdown_surrogate"):
        loss = drawdown_surrogate(
            rp, cost, init_equity=spec.init_equity, tau=spec.tau, eps=spec.eps
        )
    else:
        raise ValueError(
            "Unknown objective name. Use 'neg_log_wealth', 'neg_sharpe', or 'drawdown'."
        )

    # Regularizers
    if (spec.weight_turnover != 0.0) or (spec.weight_leverage != 0.0):
        loss = loss + regularizers(
            weights, l1_turnover=spec.weight_turnover, l1_leverage=spec.weight_leverage
        )

    return loss

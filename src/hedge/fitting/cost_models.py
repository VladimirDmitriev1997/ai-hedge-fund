# cost_models.py
"""
Reusable trading **cost models** for portfolio backtests and fitting.

Design
------
- Weight-centric: inputs are *portfolio weights* W (rows sum ~ 1 incl. CASH).
- Modular: each function returns *per-period cost rates* (same units as returns)
  so they can be plugged into backtests and optimization losses.

Conventions
-----------
Shapes (time-major):
- W: (T, M)  — row t is the allocation over assets (incl. CASH if used).
- All costs are returned as (T,) **rates** to be subtracted from portfolio returns.
- Basis points (bps) → multiply by 1e-4 to get rates.

Notes
-----
- If Torch is available and inputs are tensors, all math runs in Torch (autodiff-ready).
- These models depend only on weights (and optional per-asset coefficients), so they
  can be used with or without explicit prices/volumes.
"""


from typing import Any, Optional, Tuple, Union
from hedge.fitting.torch_utils import *
import numpy as np

try:
    import torch
    from torch import Tensor as TorchTensor

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    TorchTensor = Any  # type: ignore
    _HAS_TORCH = False


Array = Union[np.ndarray, TorchTensor]

__all__ = [
    # primitives
    "turnover_matrix",
    "l1_turnover_cost",
    "power_turnover_cost",
    "holding_short_borrow_cost",
    "leverage_funding_cost",
    "per_asset_funding_cost",
    # combos
    "sum_costs",
]


# ---------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------


def turnover_matrix(
    weights: Array,
    *,
    initial_weights: Optional[Array] = None,
) -> Array:
    """
    Construct per-period **changes in weights** ΔW_t (absolute turnover primitive).

    delta W_0 = W_0 - W_{-1}  (use `initial_weights` or assume 0)
    delta W_t = W_t - W_{t-1} for t>=1

    Parameters
    ----------
    weights : Array, shape (T, M)
        Full allocation matrix (rows ≈ simplex incl. CASH).
    initial_weights : Array or None, default None
        Baseline for the first step. If None, assume zeros of shape (M,).

    Returns
    -------
    Array, shape (T, M)
        delta W per period
    """
    W = as_array(weights)
    if W.ndim != 2:
        raise ValueError("weights must be 2D: (T, M)")
    T, M = W.shape

    if initial_weights is None:
        w_prev = (
            torch.zeros((1, M), dtype=W.dtype, device=W.device)  # type: ignore[attr-defined]
            if is_torch(W)
            else np.zeros((1, M), dtype=W.dtype)
        )
    else:
        w_prev = as_array(initial_weights)
        if w_prev.ndim == 1:
            w_prev = w_prev.reshape(1, -1)
        if w_prev.shape != (1, M):
            raise ValueError("initial_weights must have shape (M,) or (1, M)")

    if is_torch(W):
        dw = torch.vstack((W[0:1, :] - w_prev, W[1:, :] - W[:-1, :]))  # type: ignore
    else:
        dw = np.vstack((W[0:1, :] - w_prev, W[1:, :] - W[:-1, :]))
    return dw


def l1_turnover_cost(
    weights: Array,
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    per_asset_multipliers: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
) -> Array:
    """
    Linear (L1) turnover cost in rates, to subtract from returns.

    Model
    -----
    cost_t = sum_j  |delta w_{t,j}| * κ_{t,j}
      where κ_{t,j} = (fee_bps + slippage_bps)/1e4 * multiplier_{t,j}

    Parameters
    ----------
    weights : Array, shape (T, M)
        Portfolio weights.
    fee_bps : float, default 0.0
        Proportional fee per trade (basis points).
    slippage_bps : float, default 0.0
        Extra slippage per trade (basis points).
    per_asset_multipliers : Array or None, default None
        Optional scale (broadcast to (T, M)), e.g., spreads, tiers, vol.
    initial_weights : Array or None, default None
        Baseline weights for the first turnover step.

    Returns
    -------
    Array, shape (T,)
        Per-period cost **rates**.
    """
    W = as_array(weights)
    T, M = W.shape
    dw = _abs(turnover_matrix(W, initial_weights=initial_weights))

    # κ scaling
    kappa = (float(fee_bps) + float(slippage_bps)) / 1e4
    if kappa == 0.0 and per_asset_multipliers is None:
        return _zeros_like(W[:, 0])

    if per_asset_multipliers is None:
        per_asset = dw * kappa
    else:
        mult = as_array(per_asset_multipliers)
        # broadcast to (T, M)
        if is_torch(mult):
            if mult.ndim == 1:
                if mult.numel() == M:
                    mult = mult.view(1, -1).repeat(T, 1)
                elif mult.numel() == T:
                    mult = mult.view(-1, 1).repeat(1, M)
                else:
                    raise ValueError("per_asset_multipliers 1D must match M or T.")
        else:
            mult = np.asarray(mult)
            if mult.ndim == 1:
                if mult.size == M:
                    mult = np.tile(mult.reshape(1, M), (T, 1))
                elif mult.size == T:
                    mult = np.tile(mult.reshape(T, 1), (1, M))
                else:
                    raise ValueError("per_asset_multipliers 1D must match M or T.")
        per_asset = dw * (kappa * mult)

    return _sum(per_asset, axis=1)


def power_turnover_cost(
    weights: Array,
    *,
    coeff: float = 0.0,
    exponent: float = 1.5,
    per_asset_coeffs: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
) -> Array:
    """
    Power-law turnover cost |delta w|^p (p in [1,2]) — a proxy for nonlinear impact.

    Model
    -----
    cost_t = sum_j  c_{t,j} * |Δw_{t,j}|^{p}

    Parameters
    ----------
    weights : Array, shape (T, M)
        Portfolio weights.
    coeff : float, default 0.0
        Global coefficient (per period).
    exponent : float, default 1.5
        Power p; choose 1 for L1, >1 for convex (impact-like).
    per_asset_coeffs : Array or None, default None
        Optional (T,M) or broadcastable coefficients c_{t,j}.
    initial_weights : Array or None, default None
        Baseline weights for first step.

    Returns
    -------
    Array, shape (T,)
        Per-period cost rates.
    """
    if exponent < 1.0:
        raise ValueError("exponent must be >= 1.0")
    W = as_array(weights)
    dw = _abs(turnover_matrix(W, initial_weights=initial_weights))
    # |Δw|^p
    if is_torch(dw):
        mag = torch.pow(dw, exponent)  # type: ignore
    else:
        mag = dw**exponent

    if per_asset_coeffs is None:
        per_asset = mag * float(coeff)
    else:
        c = as_array(per_asset_coeffs)
        # rely on broadcasting
        per_asset = mag * (float(coeff) * c)
    return _sum(per_asset, axis=1)


def holding_short_borrow_cost(
    weights: Array,
    *,
    borrow_bps: float = 0.0,
    per_asset_borrow_bps: Optional[Array] = None,
) -> Array:
    """
    Borrow fee on short legs only (holding cost), in rates.

    Model
    -----
    cost_t = sum_j  1e-4 * borrow_bps_{t,j} * max(0, -w_{t,j})

    Parameters
    ----------
    weights : Array, shape (T, M)
        Portfolio weights (incl. CASH column if used).
    borrow_bps : float, default 0.0
        Baseline borrow fee in bps.
    per_asset_borrow_bps : Array or None, default None
        Optional (T,M) or broadcastable per-asset borrow bps.

    Returns
    -------
    Array, shape (T,)
        Per-period cost rates.
    """
    W = as_array(weights)
    short_exposure = _maximum(0.0, -W)  # only pay on shorts
    if per_asset_borrow_bps is None:
        per_asset = short_exposure * (float(borrow_bps) / 1e4)
    else:
        bps = as_array(per_asset_borrow_bps)
        per_asset = short_exposure * (bps / 1e4)
    return _sum(per_asset, axis=1)


def leverage_funding_cost(
    weights: Array,
    *,
    funding_bps: float = 0.0,
) -> Array:
    """
    Leverage funding cost based on gross exposure beyond 1.

    Model
    -----
    gross_t = sum_j |w_{t,j}|
    excess_t = max(0, gross_t - 1)
    cost_t = (funding_bps/1e4) * excess_t

    Parameters
    ----------
    weights : Array, shape (T, M)
        Portfolio weights.
    funding_bps : float, default 0.0
        Per-period funding rate (bps) applied to excess gross exposure.

    Returns
    -------
    Array, shape (T,)
        Per-period cost rates.
    """
    W = as_array(weights)
    gross = _sum(_abs(W), axis=1)
    excess = _maximum(0.0, gross - (1.0 if not is_torch(gross) else as_array(1.0)))
    rate = float(funding_bps) / 1e4
    return excess * rate  # type: ignore[operator]


def per_asset_funding_cost(
    weights: Array,
    funding_rates: Array,
) -> Array:
    """
    **Per-asset funding/financing** cost or income (can be signed), applied to positions.

    Model
    -----
    cost_t = sum_j  funding_{t,j} * w_{t,j}

    Examples
    --------
    - Perpetual swaps with funding (positive/negative).
    - Cash carry on stablecoin legs.

    Parameters
    ----------
    weights : Array, shape (T, M)
        Portfolio weights.
    funding_rates : Array
        (T, M) or broadcastable per-asset per-period rates (signed).

    Returns
    -------
    Array, shape (T,)
        Per-period net funding (positive = cost to subtract; negative = income).
    """
    W = as_array(weights)
    F = as_array(funding_rates)
    # broadcast multiply and sum over assets
    per_asset = W * F
    return _sum(per_asset, axis=1)


# ---------------------------------------------------------------------
# Combos
# ---------------------------------------------------------------------


def sum_costs(*components: Array) -> Array:
    """
    Sum multiple per-period cost rate vectors into a single cost vector.

    Parameters
    ----------
    components : Array
        Any number of (T,) arrays/tensors.

    Returns
    -------
    Array, shape (T,)
        Elementwise sum (backend preserved).
    """
    if len(components) == 0:
        raise ValueError("Provide at least one cost component.")
    acc = as_array(components[0])
    for c in components[1:]:
        acc = acc + as_array(c)
    return acc

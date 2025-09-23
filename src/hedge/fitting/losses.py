# losses.py
"""
Differentiable (NumPy / PyTorch) trading losses & utilities for strategy fitting.

Design
------
- Backend-agnostic: works with NumPy arrays or PyTorch tensors.
- Portfolio-centric: weights are full allocations (rows sum to 1.0), optionally with CASH.
- Modular costs: cost models are external and composable (by key or by callable).
- Objectives: log-wealth, Sharpe (excess), smooth drawdown.
- Pure numeric core; Torch path remains autograd-friendly.

Conventions
-----------
- Shapes are time-major.
  * weights: (T, M) — allocation after rebalance, applied over [t, t+1]
  * returns: (T, M?) — per-asset simple returns
      - if returns include CASH → shape (T, M)
      - if returns exclude CASH → shape (T, M-1) and pass rf
  * rf: float or (T,) or (T, 1) — per-period CASH return
- Costs are rates subtracted from returns (wealth factor: 1 + r_p - cost).

API summary
-----------
- portfolio_returns(weights, asset_returns, rf, has_cash_in_returns, cash_col_idx)
- combine_costs(weights, cost_items, cost_config) → per-period cost rates
- wealth_curve(port_ret, costs, init_equity, clip_min)
- neg_log_wealth / neg_sharpe / drawdown_surrogate
- regularizers(weights, l1_turnover, l1_leverage)
- Objective dataclass
- evaluate_objective(...) → scalar loss

"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, Callable, Sequence, Mapping, Dict

import numpy as np

# Backend helpers (single source of truth)
from hedge.utils import (
    _is_torch,
    _as_array,
    _zeros_like,
    _ones_like,
    _abs,
    _sum,
    _mean,
    _std,
    _log1p,
    _clip,
    _cumsum,
    _cumprod,
    _maximum,
    _where,
)

# Cost models registry
from hedge.fitting.cost_models import COST_MODELS  # exposes dict[str, callable]

try:
    import torch
    from torch import Tensor as TorchTensor

    _HAS_TORCH = True
except Exception:  # torch optional
    torch = None  # type: ignore
    TorchTensor = Any  # type: ignore
    _HAS_TORCH = False


Array = Union[np.ndarray, "TorchTensor"]
CostFn = Callable[..., Array]
CostItem = Union[CostFn, Tuple[CostFn, Mapping[str, Any]]]
CostConfig = Mapping[str, Mapping[str, Any]]


# ---------------------------------------------------------------------
# Portfolio returns
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
    Align shapes between weights (T,M) and returns (T,M or T,M-1) and inject rf for CASH if needed.
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
            raise ValueError("returns claim to include CASH but M != weights.shape[1].")
        return W, R

    if Mr != M - 1:
        raise ValueError("returns exclude CASH → expected shape (T, M-1).")

    R_full = _zeros_like(W)
    risky_cols = [j for j in range(M) if j != cash_idx]
    R_full[:, risky_cols] = R  # type: ignore[index]

    rf_arr = _as_array(rf)
    if _is_torch(rf_arr):
        if rf_arr.ndim == 0:
            rf_arr = rf_arr.repeat(T, 1)  # type: ignore
        elif rf_arr.ndim == 1:
            rf_arr = rf_arr.view(T, 1)  # type: ignore
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
    has_cash_in_returns: bool | None = None,
    cash_col_idx: Optional[int] = None,
) -> Array:
    """
    Per-period portfolio simple returns r_p,t = Σ_j w_{t,j} r_{t,j}.
    """
    W, R = _align_weights_and_returns(
        weights,
        asset_returns,
        has_cash_in_returns=has_cash_in_returns,
        rf=rf,
        cash_col_idx=cash_col_idx,
    )
    return _sum(W * R, axis=1)


# ---------------------------------------------------------------------
# Cost composition (modular; delegates to cost_models)
# ---------------------------------------------------------------------


def _resolve_cost_items(
    cost_items: Optional[Sequence[CostItem]] = None,
    cost_config: Optional[CostConfig] = None,
) -> Sequence[Tuple[CostFn, Mapping[str, Any]]]:
    """
    Normalize mixed specification into a list of (fn, kwargs).
    """
    resolved: list[Tuple[CostFn, Mapping[str, Any]]] = []

    if cost_items:
        for item in cost_items:
            if callable(item):
                resolved.append((item, {}))  # type: ignore[arg-type]
            else:
                fn, kwargs = item  # type: ignore[misc]
                resolved.append((fn, dict(kwargs or {})))

    if cost_config:
        for key, kwargs in cost_config.items():
            if key not in COST_MODELS:
                raise KeyError(
                    f"Unknown cost model key '{key}'. Available: {sorted(COST_MODELS.keys())}"
                )
            resolved.append((COST_MODELS[key], dict(kwargs or {})))

    return resolved


def combine_costs(
    weights: Array,
    *,
    cost_items: Optional[Sequence[CostItem]] = None,
    cost_config: Optional[CostConfig] = None,
) -> Array:
    """
    Sum multiple per-period cost rate series into a single vector of shape (T,).
    """
    specs = _resolve_cost_items(cost_items, cost_config)
    if not specs:
        W = _as_array(weights)
        return _zeros_like(W[:, 0])

    total = None
    for fn, kwargs in specs:
        c = _as_array(fn(weights=weights, **kwargs))
        total = c if total is None else (total + c)
    return _as_array(total)


# ---------------------------------------------------------------------
# Wealth curve
# ---------------------------------------------------------------------


def wealth_curve(
    port_ret: Array,
    costs: Optional[Array] = None,
    *,
    init_equity: float = 1.0,
    clip_min: float = -0.999999,
) -> Array:
    """
    Equity curve via recursion: E_{t+1} = E_t * (1 + r_{p,t} - cost_t).
    """
    r = _as_array(port_ret)
    c = _zeros_like(r) if costs is None else _as_array(costs)
    g = _clip(r - c, clip_min, np.inf)
    gross = 1.0 + g
    return _cumprod(gross, axis=0) * float(init_equity)


# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------


def neg_log_wealth(
    port_ret: Array,
    costs: Optional[Array] = None,
    *,
    eps: float = 1e-12,
) -> Array:
    """
    Negative log-wealth: L = -Σ_t log(1 + r_{p,t} - cost_t + eps).
    """
    r = _as_array(port_ret)
    c = _zeros_like(r) if costs is None else _as_array(costs)
    val = _log1p(r - c + (eps if not _is_torch(r) else _as_array(eps)))
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
    Negative Sharpe (excess): - (mean(x) / std(x)) * sqrt(annualizer), x = r_p - cost - rf.
    """
    r = _as_array(port_ret)
    c = _zeros_like(r) if costs is None else _as_array(costs)
    rf = _as_array(rf_per_period)

    if _is_torch(r):
        if rf.ndim == 0:
            rf = rf.repeat(r.shape[0])  # type: ignore
        elif rf.ndim > 1:
            rf = rf.view(-1)  # type: ignore
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
    Smooth drawdown: mean( softplus( ((P_t - E_t)/(P_t + eps))*tau ) / tau ), P_t = running max(E_t).
    Torch path uses `torch.cummax`; NumPy path uses `np.maximum.accumulate`.
    """
    eq = wealth_curve(port_ret, costs, init_equity=init_equity)

    if _is_torch(eq):
        peak = torch.cummax(eq, dim=0)[0]  # type: ignore
        rel_gap = (peak - eq) / (peak + _as_array(eps))
        soft = torch.nn.functional.softplus(rel_gap * tau) / tau  # type: ignore
        return _mean(soft)

    peak = np.maximum.accumulate(eq)
    rel_gap = (peak - eq) / (peak + eps)
    soft = np.log1p(np.exp(rel_gap * tau)) / tau
    return _mean(soft)


def regularizers(
    weights: Array,
    *,
    l1_turnover: float = 0.0,
    l1_leverage: float = 0.0,
) -> Array:
    """
    Linear penalties:
    R = λ_to * mean_t Σ_j |Δw_{t,j}|  +  λ_lev * mean_t Σ_j |w_{t,j}|.
    """
    if (l1_turnover == 0.0) and (l1_leverage == 0.0):
        W = _as_array(weights)
        return _sum(W[:, :1] * 0.0)

    W = _as_array(weights)

    if _is_torch(W):
        dw = torch.vstack((W[0:1, :], W[1:, :] - W[:-1, :]))  # type: ignore
    else:
        dw = np.vstack((W[0:1, :], W[1:, :] - W[:-1, :]))

    pen_to = float(l1_turnover) * _mean(_sum(_abs(dw), axis=1))
    pen_lev = float(l1_leverage) * _mean(_sum(_abs(W), axis=1))
    return pen_to + pen_lev


# ---------------------------------------------------------------------
# Objective wrapper
# ---------------------------------------------------------------------


@dataclass
class Objective:
    """
    Container for objective settings.

    name : {"neg_log_wealth", "neg_sharpe", "drawdown"}
    annualizer : float (Sharpe)
    rf_per_period : float | Array (Sharpe)
    init_equity : float (drawdown)
    eps : float
    tau : float (drawdown)
    weight_turnover : float (regularizer)
    weight_leverage : float (regularizer)
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
    # Modular costs
    cost_items: Optional[Sequence[CostItem]] = None,
    cost_config: Optional[CostConfig] = None,
    # Legacy convenience (auto "turnover" if any provided)
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    per_asset_multipliers: Optional[Array] = None,
    initial_weights: Optional[Array] = None,
    # Loss settings
    spec: Objective = Objective(),
) -> Array:
    """
    End-to-end objective: weights + returns → r_p, compose costs → base loss → add regularizers.

    Costs (unified)
    ---------------
    - Preferred: `cost_config` with keys from COST_MODELS, or `cost_items` as callables.
    - Legacy: fee_bps/slippage_bps/per_asset_multipliers/initial_weights → auto "turnover".
    """
    rp = portfolio_returns(
        weights,
        asset_returns,
        rf=rf,
        has_cash_in_returns=has_cash_in_returns,
        cash_col_idx=cash_col_idx,
    )

    auto_cfg: Dict[str, Mapping[str, Any]] = {}
    if (
        (fee_bps != 0.0)
        or (slippage_bps != 0.0)
        or (per_asset_multipliers is not None)
        or (initial_weights is not None)
    ):
        auto_cfg["turnover"] = {
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "per_asset_multipliers": per_asset_multipliers,
            "initial_weights": initial_weights,
        }

    merged_cfg: Dict[str, Mapping[str, Any]] = {}
    if cost_config:
        merged_cfg.update(cost_config)
    if auto_cfg:
        merged_cfg = {**auto_cfg, **merged_cfg}

    cost = combine_costs(
        weights, cost_items=cost_items, cost_config=(merged_cfg or None)
    )

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
            "Unknown objective: {'neg_log_wealth','neg_sharpe','drawdown'} expected."
        )

    if (spec.weight_turnover != 0.0) or (spec.weight_leverage != 0.0):
        loss = loss + regularizers(
            weights, l1_turnover=spec.weight_turnover, l1_leverage=spec.weight_leverage
        )
    return loss

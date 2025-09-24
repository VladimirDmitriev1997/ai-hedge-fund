# losses.py
"""
Differentiable (NumPy / PyTorch) trading losses & utilities for strategy fitting.

Conventions
-----------
- Shapes are time-major.
  * weights: (T, M) — applied over [t, t+1]
  * returns: (T, M?) — per-asset simple returns
      - if returns include CASH → shape (T, M)
      - if returns exclude CASH → shape (T, M-1) and pass `rf`
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

import warnings
import numpy as np

# Backend helpers (single source of truth)
from hedge.fitting.torch_utils import (
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

from hedge.portfolio import _align_weights_and_returns, portfolio_returns


# Cost models registry (expected to be defined in cost_models.py)
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
# Cost composition (modular; delegates to cost_models)
# ---------------------------------------------------------------------


def _resolve_cost_items(
    cost_items: Optional[Sequence[CostItem]] = None,
    cost_config: Optional[CostConfig] = None,
) -> Sequence[Tuple[CostFn, Mapping[str, Any]]]:
    """
    Normalize mixed specification into a list of (fn, kwargs).

    Parameters
    ----------
    cost_items : sequence of callables or (callable, kwargs), optional
        Explicit list of cost model callables with optional kwargs.
        Each callable must return per-period cost rates (T,).
    cost_config : mapping, optional
        Dict-based registry use: {key: kwargs} where key ∈ COST_MODELS.

    Returns
    -------
    list[tuple[callable, dict]]
        Normalized list of (fn, kwargs).
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
    Sum multiple per-period cost rate series into a single vector (T,).

    Parameters
    ----------
    weights : Array
        Portfolio weights, shape (T, M). Passed to cost model callables.
    cost_items : sequence, optional
        Explicit list of (fn, kwargs) or fn returning (T,).
    cost_config : mapping, optional
        Dict-based specification via COST_MODELS registry.

    Returns
    -------
    Array
        Total per-period cost rates, shape (T,).
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
    Build equity curve E via E_{t+1} = E_t * (1 + r_{p,t} - cost_t), with clipping on per-period gains.

    Parameters
    ----------
    port_ret : Array
        Portfolio simple returns r_{p,t}, shape (T,).
    costs : Array, optional
        Per-period cost rates, shape (T,). If None, assumed zero.
    init_equity : float, optional
        Initial equity multiplier E_0. Default 1.0.
    clip_min : float, optional
        Lower clip on (r_{p,t} - cost_t) before 1+·, to avoid crossing ≤ -1.

    Returns
    -------
    Array
        Equity curve E_t, shape (T,). (Starts at 1+g_0, so first element reflects first period.)
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

    Parameters
    ----------
    port_ret : Array
        Portfolio simple returns r_{p,t}, shape (T,).
    costs : Array, optional
        Per-period cost rates, shape (T,). If None, assumed zero.
    eps : float, optional
        Numerical stabilizer inside log1p.

    Returns
    -------
    Array
        Scalar loss (0-d array / tensor).
    """
    r = _as_array(port_ret)
    c = _zeros_like(r) if costs is None else _as_array(costs)
    eps_arr = _as_array(eps) if _is_torch(r) else float(eps)
    val = _log1p(r - c + eps_arr)
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
    Negative Sharpe (excess):  - (mean(x) / std(x)) * sqrt(annualizer),  x = r_p - cost - rf.

    Parameters
    ----------
    port_ret : Array
        Portfolio simple returns r_{p,t}, shape (T,).
    costs : Array, optional
        Per-period cost rates, shape (T,). If None, assumed zero.
    annualizer : float
        Scaling factor for Sharpe (e.g., bars_per_unit like 365 for daily).
    rf_per_period : float or Array, optional
        Risk-free per period (same frequency as `port_ret`). Scalar or (T,).
    eps : float, optional
        Numerical floor for std to avoid division by zero.

    Returns
    -------
    Array
        Scalar loss (0-d array / tensor).
    """
    r = _as_array(port_ret)
    c = _zeros_like(r) if costs is None else _as_array(costs)
    rf = _as_array(rf_per_period)

    if _is_torch(r):
        if rf.ndim == 0:
            rf = rf.repeat(r.shape[0])  # type: ignore[attr-defined]
        elif rf.ndim > 1:
            rf = rf.view(-1)  # type: ignore[attr-defined]
    else:
        rf = np.asarray(rf)
        if rf.ndim == 0:
            rf = np.full_like(r, float(rf))
        elif rf.ndim > 1:
            rf = rf.reshape(-1)  # ensure shape (T,)

    x = r - c - rf
    mu = _mean(x)
    sd = _std(x)  # ddof=0; backend handles keepdims=False
    sd_safe = _maximum(sd, _as_array(eps) if _is_torch(x) else float(eps))
    sharpe = mu / sd_safe * (annualizer**0.5)
    return -sharpe


def drawdown_surrogate(
    port_ret: Array,
    costs: Optional[Array] = None,
    *,
    init_equity: float = 1.0,
    tau: float = 10.0,
    eps: float = 1e-12,
    smooth_max_tau: Optional[float] = None,
) -> Array:
    """
    Smooth drawdown surrogate.

    By default uses a hard running-max P_t = max_{s≤t} E_s, then softplus on relative gap.
    If `smooth_max_tau` is provided (>0), uses a soft running max:
        P_t ≈ (1/τ_m) log Σ_{s≤t} exp(τ_m E_s)

    Parameters
    ----------
    port_ret : Array
        Portfolio simple returns r_{p,t}, shape (T,).
    costs : Array, optional
        Per-period cost rates, shape (T,). If None, assumed zero.
    init_equity : float, optional
        Initial equity multiplier E_0. Default 1.0.
    tau : float, optional
        Temperature for softplus on relative drawdown (larger → sharper).
    eps : float, optional
        Numerical stabilizer for division by peak.
    smooth_max_tau : float, optional
        Temperature τ_m for soft running max; if None, uses hard running max.

    Returns
    -------
    Array
        Scalar loss (0-d array / tensor).
    """
    eq = wealth_curve(port_ret, costs, init_equity=init_equity)

    if smooth_max_tau is not None and smooth_max_tau > 0:
        # Soft running max via cumulative log-sum-exp (O(T^2); opt-in only).
        tau_m = float(smooth_max_tau)
        if _is_torch(eq):
            T = eq.shape[0]
            peaks = []
            for t in range(T):
                # logsumexp over 0..t
                v = torch.logsumexp(eq[: t + 1] * tau_m, dim=0) / tau_m  # type: ignore[attr-defined]
                peaks.append(v)
            peak = torch.stack(peaks, dim=0)
            rel_gap = (peak - eq) / (peak + _as_array(eps))
            soft = torch.nn.functional.softplus(rel_gap * tau) / tau  # type: ignore[attr-defined]
            return _mean(soft)
        else:
            T = eq.shape[0]
            peak = np.empty_like(eq)
            for t in range(T):
                # logsumexp over 0..t (numpy)
                z = np.log(np.sum(np.exp(eq[: t + 1] * tau_m))) / tau_m
                peak[t] = z
            rel_gap = (peak - eq) / (peak + float(eps))
            soft = np.log1p(np.exp(rel_gap * tau)) / tau
            return _mean(soft)

    # Hard running max (default path)
    if _is_torch(eq):
        peak = torch.cummax(eq, dim=0)[0]  # type: ignore[attr-defined]
        rel_gap = (peak - eq) / (peak + _as_array(eps))
        soft = torch.nn.functional.softplus(rel_gap * tau) / tau  # type: ignore[attr-defined]
        return _mean(soft)

    peak = np.maximum.accumulate(eq)
    rel_gap = (peak - eq) / (peak + float(eps))
    soft = np.log1p(np.exp(rel_gap * tau)) / tau
    return _mean(soft)


def regularizers(
    weights: Array,
    *,
    l1_turnover: float = 0.0,
    l1_leverage: float = 0.0,
) -> Array:
    """
    Linear penalties (regularization, not real commissions).

    R = λ_to * mean_t Σ_j |Δw_{t,j}|  +  λ_lev * mean_t Σ_j |w_{t,j}|

    Notes
    -----
    - This is a *regularizer*. Real transaction costs (incl. conventional ½·Σ|Δw|)
      must be modeled via `cost_models` and passed through `combine_costs`.
    - We set Δw_0 = 0 (no penalty on the first row) unless you explicitly model
      initial weights in a cost model.

    Parameters
    ----------
    weights : Array
        Portfolio weights, shape (T, M).
    l1_turnover : float, optional
        Coefficient λ_to for L1 penalty on turnover.
    l1_leverage : float, optional
        Coefficient λ_lev for L1 penalty on leverage (gross exposure).

    Returns
    -------
    Array
        Scalar penalty (0-d array / tensor).
    """
    if (l1_turnover == 0.0) and (l1_leverage == 0.0):
        W = _as_array(weights)
        return _sum(W[:, :1] * 0.0)

    W = _as_array(weights)

    # Δw with Δw_0 = 0 (no penalty at t=0)
    if _is_torch(W):
        dw = torch.zeros_like(W)  # type: ignore[attr-defined]
        dw[1:, :] = W[1:, :] - W[:-1, :]
    else:
        dw = np.zeros_like(W)
        dw[1:, :] = W[1:, :] - W[:-1, :]

    pen_to = float(l1_turnover) * _mean(_sum(_abs(dw), axis=1))
    pen_lev = float(l1_leverage) * _mean(_sum(_abs(W), axis=1))
    return pen_to + pen_lev


# ---------------------------------------------------------------------
# Objective wrapper
# ---------------------------------------------------------------------


@dataclass
class Objective:
    """
    Objective settings.

    name : {"neg_log_wealth", "neg_sharpe", "drawdown"}
        Which base loss to compute.
    annualizer : float (Sharpe)
        Scaling factor for Sharpe (e.g., bars_per_unit like 365 for daily).
    rf_per_period : float or Array (Sharpe)
        Risk-free per period (scalar or (T,)).
    init_equity : float (drawdown)
        Initial equity multiplier (E_0).
    eps : float
        Numerical stabilizer for log/denominators.
    tau : float (drawdown)
        Temperature for softplus on relative drawdown.
    smooth_max_tau : float, optional (drawdown)
        Temperature τ_m for soft running max. None → hard running max.
    weight_turnover : float (regularizer)
        λ for L1 turnover penalty (regularization, not real commissions).
    weight_leverage : float (regularizer)
        λ for L1 leverage penalty.

    Notes
    -----
    - Real costs should be passed via `cost_config`/`cost_items` and modeled in
      `cost_models` (where the conventional ½·Σ|Δw| belongs).
    """

    name: str = "neg_log_wealth"
    annualizer: float = 365.0
    rf_per_period: float | Array = 0.0
    init_equity: float = 1.0
    eps: float = 1e-12
    tau: float = 10.0
    smooth_max_tau: Optional[float] = None
    weight_turnover: float = 0.0
    weight_leverage: float = 0.0


def evaluate_objective(
    weights: Array,
    asset_returns: Array,
    *,
    rf: float | Array = 0.0,
    has_cash_in_returns: Optional[bool] = None,
    cash_col_idx: Optional[int] = None,
    # Modular costs
    cost_items: Optional[Sequence[CostItem]] = None,
    cost_config: Optional[CostConfig] = None,
    # (Auto "turnover" if any provided)
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
    Preferred: use `cost_config` with keys from COST_MODELS, or `cost_items` as callables.
    Legacy: fee_bps / slippage_bps / per_asset_multipliers / initial_weights → auto "turnover".

    Parameters
    ----------
    weights : Array
        Portfolio weights, shape (T, M).
    asset_returns : Array
        Per-asset simple returns, shape (T, M) if includes CASH, else (T, M-1).
    rf : float or Array, optional
        Per-period risk-free (CASH) return for `portfolio_returns` if returns exclude CASH.
    has_cash_in_returns : bool, optional
        Whether `asset_returns` already includes CASH. If None, inferred by shape.
    cash_col_idx : int, optional
        Index of CASH column in `weights`. Defaults to last column.
    cost_items : sequence, optional
        Explicit list of cost callables returning (T,).
    cost_config : mapping, optional
        Dict-based config via COST_MODELS.
    fee_bps, slippage_bps, per_asset_multipliers, initial_weights : legacy, optional
        If any are provided, an auto "turnover" cost is added to `cost_config`.
    spec : Objective
        Base loss configuration and regularizer weights.

    Returns
    -------
    Array
        Scalar loss (0-d array / tensor).

    Notes
    -----
    If both real turnover costs (in `cost_config`) and `spec.weight_turnover>0`
    are set, this may constitute *double counting*; a warning is emitted.
    """
    rp = portfolio_returns(
        weights,
        asset_returns,
        rf=rf,
        has_cash_in_returns=has_cash_in_returns,
        cash_col_idx=cash_col_idx,
    )

    # Build merged cost configuration (legacy auto-turnover → "turnover" key)
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

    # Warn on potential double counting (real turnover + L1-turnover regularizer)
    if (spec.weight_turnover != 0.0) and ("turnover" in merged_cfg):
        warnings.warn(
            "Both real turnover costs (cost_config['turnover']) and L1-turnover "
            "regularizer (spec.weight_turnover) are active. This may double-count costs.",
            stacklevel=2,
        )

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
            rp,
            cost,
            init_equity=spec.init_equity,
            tau=spec.tau,
            eps=spec.eps,
            smooth_max_tau=spec.smooth_max_tau,
        )
    else:
        raise ValueError(
            "Unknown objective: expected one of {'neg_log_wealth','neg_sharpe','drawdown'}."
        )

    if (spec.weight_turnover != 0.0) or (spec.weight_leverage != 0.0):
        loss = loss + regularizers(
            weights, l1_turnover=spec.weight_turnover, l1_leverage=spec.weight_leverage
        )
    return loss

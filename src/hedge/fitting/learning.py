# learning.py
"""
Learning and optimization utilities for strategies.

Design
------
- Single entry-point `fit_strategy(...)` that:
  * takes a strategy instance (dataclass-based or torch-trainable),
  * consumes a training DataFrame `df` (features/prices) and an asset returns
    matrix `asset_returns` aligned by index (and by asset columns),
  * builds an objective from `loss_name|Objective`, `cost_config`, RF conventions,
  * runs optimization according to `mode={"grid","random","gradient"}`,
  * returns a `FitResult` with best params, best weights, and history.

Conventions
-----------
- Shapes are time-major.
  * weights: (T, M) — applied over [t, t+1]
  * returns: (T, M?) — per-asset simple returns; if CASH is not included,
    pass `has_cash_in_returns=False` and provide `rf_annual` (we inject rf).
- Risk-free convention: pass `rf_annual` + `bars_per_unit`; internally we use
  `rf_per_bar = (1 + rf_annual)^(1 / bars_per_unit) - 1`.
- Costs: pass *real trading costs* via `cost_config` (unified with COST_MODELS).
  Regularizers (e.g., L1 on turnover/leverage) are configured in the Objective.
- Gradient mode: requires a torch-capable strategy exposing
  `forward_torch(df) -> TorchTensor (T,M)` and `torch_parameters() -> Iterable[Parameter]`.

Notes
-----
- This module is agnostic to cross-validation; use `cv.py`/`evaluation.py` later.
- For non-differentiable hyperparameters (e.g., integer windows) prefer "grid" or "random".
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Callable,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    import torch
import itertools
import math
import random

import numpy as np
import pandas as pd

from hedge.fitting.losses import Objective, evaluate_objective
from hedge.portfolio import CASH_COL  # unified CASH column name


try:
    import torch
    from torch import Tensor as TorchTensor

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    TorchTensor = Any  # type: ignore
    _HAS_TORCH = False


# ---------------------------------------------------------------------
# Public API (exports)
# ---------------------------------------------------------------------

__all__ = [
    "LearningConfig",
    "OptimizationConfig",
    "FitResult",
    "fit_strategy",
]


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass
class LearningConfig:
    """
    Configuration of the learning objective and data alignment.

    Parameters
    ----------
    rf_annual : float, default 0.0
        Risk-free rate per chosen unit (e.g., annual). Will be converted to per-bar.
    bars_per_unit : float, default 252.0
        Number of bars per the chosen unit for rf/per-bar conversions.
    has_cash_in_returns : bool or None, default None
        If None, inferred from `asset_returns` shape vs weights (M vs M-1).
    cash_col_idx : int or None, default None
        Index of CASH column within weights/returns (default: last column).
    cost_config : Mapping[str, Mapping[str, Any]] or None
        Unified trading cost configuration keyed by cost-model names (COST_MODELS).
    fee_bps : float, default 0.0
        Legacy convenience: proportional fee in bps (auto-turnover).
    slippage_bps : float, default 0.0
        Legacy convenience: slippage in bps (auto-turnover).
    per_asset_multipliers : Array-like or None, default None
        Optional per-asset multipliers for legacy turnover cost.
    initial_weights : Array-like or None, default None
        Baseline weights for turnover at t=0 (legacy path).
    objective : Objective or str, default "neg_log_wealth"
        Objective settings (name/eps/annualizer etc.). If str, default settings
        are used and overridden where applicable.
    """

    rf_annual: float = 0.0
    bars_per_unit: float = 252.0
    has_cash_in_returns: Optional[bool] = None
    cash_col_idx: Optional[int] = None

    # Costs (preferred: cost_config). Legacy convenience kept for b/c.
    cost_config: Optional[Mapping[str, Mapping[str, Any]]] = None
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    per_asset_multipliers: Optional[Any] = None
    initial_weights: Optional[Any] = None

    # Objective
    objective: Union[str, Objective] = "neg_log_wealth"

    def rf_per_bar(self) -> float:
        """
        Convert rf per unit to per-bar: (1 + rf_annual)^(1/bars_per_unit) - 1.

        Returns
        -------
        float
            Per-bar risk-free rate.
        """
        return (1.0 + float(self.rf_annual)) ** (1.0 / float(self.bars_per_unit)) - 1.0


@dataclass
class OptimizationConfig:
    """
    Optimization mode and hyperparameters.

    Parameters
    ----------
    mode : {'grid','random','gradient'}
        Choice of optimization algorithm.
    # Grid search
    param_grid : Mapping[str, Sequence[Any]] or None
        Grid of parameter values for exhaustive search (cartesian product).
    # Random search
    param_distributions : Mapping[str, Any] or None
        Distributions for random sampling. Supported:
          - list/tuple of discrete values (random choice),
          - tuple(lo, hi) → uniform float in [lo, hi],
          - ('loguniform', lo, hi) → log-uniform float in [lo, hi].
    n_iter : int, default 50
        Number of samples for random search.
    random_state : Optional[int], default None
        Seed for reproducibility.
    # Gradient (torch)
    optimizer : {'adam','sgd','lbfgs'}, default 'adam'
        Optimizer for gradient mode.
    lr : float, default 1e-2
        Learning rate.
    weight_decay : float, default 0.0
        L2 regularization for the optimizer.
    epochs : int, default 200
        Number of epochs/steps.
    device : str or None, default None
        Torch device spec ('cpu'/'cuda:0'); None → auto.
    """

    mode: str = "grid"

    # Grid
    param_grid: Optional[Mapping[str, Sequence[Any]]] = None

    # Random
    param_distributions: Optional[Mapping[str, Any]] = None
    n_iter: int = 50
    random_state: Optional[int] = None

    # Gradient
    optimizer: str = "adam"
    lr: float = 1e-2
    weight_decay: float = 0.0
    epochs: int = 200
    device: Optional[str] = None


@dataclass
class FitResult:
    """
    Result of fitting a strategy on a dataset.

    Attributes
    ----------
    best_params : Dict[str, Any]
        Best parameter set found.
    best_loss : float
        Minimum objective value (lower is better).
    best_weights : pd.DataFrame
        Weights produced by the best parameter set (aligned to df.index).
    history : List[Tuple[Dict[str, Any], float]]
        Sequence of (params, loss) pairs evaluated during the search.
    updated_strategy : Any
        Strategy instance with best parameters applied (reference to input object).
    meta : Dict[str, Any]
        Misc metadata (mode, epochs, seeds, etc.).
    """

    best_params: Dict[str, Any]
    best_loss: float
    best_weights: pd.DataFrame
    history: List[Tuple[Dict[str, Any], float]]
    updated_strategy: Any
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _ensure_returns(returns: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Ensure an asset returns matrix as a NumPy array (time-major).

    Parameters
    ----------
    returns : pd.DataFrame or np.ndarray
        Per-asset simple returns (T,M?) aligned to weights.

    Returns
    -------
    np.ndarray
        Dense array with shape (T, M?) suitable for objective evaluation.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.to_numpy()
    arr = np.asarray(returns)
    if arr.ndim != 2:
        raise ValueError("asset_returns must be 2D (T, M or M-1).")
    return arr


def _infer_bars_per_unit(idx: pd.DatetimeIndex) -> float:
    """Infer bars-per-unit (≈ per year) from median spacing if cfg value is <= 0."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return 252.0
    deltas = idx.to_series().diff().dropna().dt.total_seconds()
    if len(deltas) == 0:
        return 252.0
    med = float(deltas.median())
    if med <= 0:
        return 252.0
    sec_per_year = 365.0 * 24.0 * 3600.0
    return sec_per_year / med


def _extract_rf_from_df(
    df: pd.DataFrame, fallback_per_bar: float
) -> Union[float, np.ndarray]:
    """
    Try to find a per-bar RF series in df; otherwise return scalar fallback.

    Supported columns:
      - flat: 'rf', 'risk_free', 'cash_return'
      - MultiIndex: ('CASH','return'), ('RF','per_bar')
    """
    # MultiIndex first
    if isinstance(df.columns, pd.MultiIndex):
        for cand in [("CASH", "return"), ("RF", "per_bar")]:
            if cand in df.columns:
                s = pd.to_numeric(df[cand], errors="coerce").fillna(0.0).to_numpy()
                return s
    else:
        for name in ("rf", "risk_free", "cash_return"):
            if name in df.columns:
                s = pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy()
                return s
    # fallback scalar
    return float(fallback_per_bar)


def _objective_from_config(cfg: LearningConfig) -> Objective:
    """
    Build an Objective from a string or pass-through.

    Parameters
    ----------
    cfg : LearningConfig
        Learning configuration.

    Returns
    -------
    Objective
        Objective instance with RF/annualizer aligned.
    """
    if isinstance(cfg.objective, Objective):
        obj = cfg.objective
    else:
        name = str(cfg.objective).lower()
        obj = Objective(name=name)

    # Set annualizer for Sharpe; RF per-period will be set downstream
    if obj.name.lower() == "neg_sharpe":
        # Allow dynamic inference when cfg.bars_per_unit <= 0 (handled in fit entry)
        obj.annualizer = float(cfg.bars_per_unit)
    return obj


def _eval_loss_for_weights(
    W: Union[pd.DataFrame, np.ndarray],
    asset_returns: Union[pd.DataFrame, np.ndarray],
    cfg: LearningConfig,
    rf_per_bar: Optional[Union[float, np.ndarray]] = None,
) -> float:
    """
    Evaluate objective value for provided weights.

    Parameters
    ----------
    W : pd.DataFrame or np.ndarray
        Weight matrix (T, M), includes CASH column.
    asset_returns : pd.DataFrame or np.ndarray
        Per-asset returns (T, M or T, M-1).
    cfg : LearningConfig
        Learning configuration including costs and objective.
    rf_per_bar : float or (T,), optional
        If provided, overrides cfg.rf_per_bar() for this evaluation.

    Returns
    -------
    float
        Scalar objective value (lower is better).
    """
    R = _ensure_returns(asset_returns)
    W_arr = W.to_numpy() if isinstance(W, pd.DataFrame) else np.asarray(W)

    # RF override (vector or scalar)
    rf_used = cfg.rf_per_bar() if rf_per_bar is None else rf_per_bar

    obj = _objective_from_config(cfg)
    # For Sharpe, ensure the same rf_per_period is used (can be scalar or vector)
    if obj.name.lower() == "neg_sharpe":
        obj.rf_per_period = rf_used  # may be float or ndarray

    loss_val = evaluate_objective(
        weights=W_arr,
        asset_returns=R,
        rf=rf_used,
        has_cash_in_returns=cfg.has_cash_in_returns,
        cash_col_idx=cfg.cash_col_idx,
        cost_items=None,
        cost_config=cfg.cost_config,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        per_asset_multipliers=cfg.per_asset_multipliers,
        initial_weights=cfg.initial_weights,
        spec=obj,
    )
    # evaluate_objective returns backend scalar; cast to float
    if hasattr(loss_val, "item"):
        return float(loss_val.item())
    return float(loss_val)


# --- add these small helpers somewhere above train_on_csv in run.py ---


def _extract_single_asset_ohlcv(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    From a MultiIndex (symbol, field) frame, extract one asset into a flat OHLCV frame.
    """
    fields = ["open", "high", "low", "close", "volume"]
    need = [(asset, f) for f in fields]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(
            f"{asset}: not found in portfolio frame (need {asset}.close or OHLCV)."
        )
    out = pd.DataFrame({f: df[(asset, f)] for f in fields}, index=df.index)
    # numeric + clean
    for c in fields:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(how="any").astype("float64")
    return out


def _returns_with_cash(
    returns_risky: pd.DataFrame, cash_col: str = CASH_COL
) -> pd.DataFrame:
    """
    Append a zero return CASH column and ensure CASH is last.
    """
    R = returns_risky.copy()
    R[cash_col] = 0.0
    cols = [c for c in R.columns if c != cash_col] + [cash_col]
    return R.loc[:, cols].astype("float64")


def _iter_grid(param_grid: Mapping[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    """
    Cartesian product over parameter grid.

    Parameters
    ----------
    param_grid : Mapping[str, Sequence[Any]]
        Dict of parameter name → sequence of candidate values.

    Returns
    -------
    Iterable[Dict[str, Any]]
        Generator of param dicts.
    """
    if not param_grid:
        yield {}
        return
    keys = list(param_grid.keys())
    vals = [list(v) for v in param_grid.values()]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def _rng_choice(rng: random.Random, spec: Any) -> Any:
    """
    Sample a value from a simple distribution spec.

    Supported specs
    ---------------
    - list/tuple of discrete values → random choice
    - (lo, hi) → uniform float in [lo, hi]
    - ('loguniform', lo, hi) → log-uniform in [lo, hi] for positive lo,hi

    Parameters
    ----------
    rng : random.Random
        RNG instance.
    spec : Any
        Distribution specification.

    Returns
    -------
    Any
        Sampled value.
    """
    if isinstance(spec, (list, tuple)) and len(spec) >= 2 and isinstance(spec[0], str):
        mode = spec[0].lower()
        if mode == "loguniform":
            _, lo, hi = spec
            lo = float(lo)
            hi = float(hi)
            if lo <= 0 or hi <= 0:
                raise ValueError("loguniform bounds must be positive")
            u = rng.random()
            return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
            lo, hi = spec
            return rng.uniform(float(lo), float(hi))
        # discrete
        return rng.choice(list(spec))
    # single value → pass-through
    return spec


def _iter_random(
    param_distributions: Mapping[str, Any], n_iter: int, seed: Optional[int]
) -> Iterable[Dict[str, Any]]:
    """
    Randomly sample parameter configurations.

    Parameters
    ----------
    param_distributions : Mapping[str, Any]
        Dict of name → distribution spec.
    n_iter : int
        Number of draws.
    seed : int or None
        RNG seed.

    Returns
    -------
    Iterable[Dict[str, Any]]
        Generator of param dicts.
    """
    rng = random.Random(seed)
    keys = list(param_distributions.keys()) if param_distributions else []
    dists = [param_distributions[k] for k in keys] if param_distributions else []
    for _ in range(max(1, int(n_iter))):
        params = {k: _rng_choice(rng, dist) for k, dist in zip(keys, dists)}
        yield params


def _torch_device(opt: OptimizationConfig) -> Optional[Any]:  # type: ignore[name-defined]
    """
    Resolve torch device for gradient mode.

    Parameters
    ----------
    opt : OptimizationConfig
        Optimization settings.

    Returns
    -------
    torch.device or None
        Selected device or None if torch is unavailable.
    """
    if not _HAS_TORCH:
        return None

    if opt.device is not None:
        return torch.device(opt.device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------


def fit_strategy(
    strategy: Any,
    df: pd.DataFrame,
    asset_returns: Union[pd.DataFrame, np.ndarray],
    *,
    learning: Optional[LearningConfig] = None,
    optimization: Optional[OptimizationConfig] = None,
) -> FitResult:
    """
    Fit a strategy on a dataset by minimizing a specified objective.

    Parameters
    ----------
    strategy : Any
        Strategy instance. Should implement:
          - `predict_weights(df) -> StrategyResult` (for grid/random modes).
        For gradient mode, additionally:
          - `forward_torch(df) -> TorchTensor (T, M)` (weights including CASH),
          - `torch_parameters() -> Iterable[torch.nn.Parameter]`.
    df : pd.DataFrame
        Training dataframe (features/prices), index aligned to returns.
    asset_returns : pd.DataFrame or np.ndarray
        Per-asset simple returns aligned to df.index.
        Shape (T,M) if returns include CASH; else (T,M-1) and RF is injected.
    learning : LearningConfig or None
        Objective/cost/RF configuration. If None, defaults are used.
    optimization : OptimizationConfig or None
        Optimization mode/hyperparameters. If None, defaults to grid with empty grid.

    Returns
    -------
    FitResult
        Best parameters and loss; best weights; evaluation history; updated strategy.

    Notes
    -----
    - This function does not perform cross-validation; use `cv.py` later.
    - If you pass legacy `fee_bps`/`slippage_bps` in `learning`, a turnover cost
      is auto-injected; avoid double-counting with explicit cost_config.
    """
    if learning is None:
        learning = LearningConfig()
    if optimization is None:
        optimization = OptimizationConfig()

    # Optional: infer bars_per_unit from data spacing if cfg value is invalid/nonpositive
    if getattr(learning, "bars_per_unit", 252.0) <= 0 and isinstance(
        df.index, pd.DatetimeIndex
    ):
        learning.bars_per_unit = _infer_bars_per_unit(df.index)

    mode = str(optimization.mode).lower()
    if mode == "grid":
        return _fit_grid(strategy, df, asset_returns, learning, optimization)
    elif mode == "random":
        return _fit_random(strategy, df, asset_returns, learning, optimization)
    elif mode == "gradient":
        return _fit_gradient(strategy, df, asset_returns, learning, optimization)
    else:
        raise ValueError(
            "optimization.mode must be one of {'grid','random','gradient'}"
        )


# ---------------------------------------------------------------------
# Implementations: grid / random
# ---------------------------------------------------------------------


def _apply_params(strategy: Any, params: Mapping[str, Any]) -> Any:
    """
    Set parameters on the strategy using a tolerant interface.

    Parameters
    ----------
    strategy : Any
        Strategy instance (dataclass-based).
    params : Mapping[str, Any]
        Parameter values.

    Returns
    -------
    Any
        Strategy with updated params (in-place if supported).

    Notes
    -----
    - Prefers `set_params(**params)` if present; otherwise setattr for each field.
    """
    if hasattr(strategy, "set_params") and callable(getattr(strategy, "set_params")):
        return strategy.set_params(**dict(params))
    # Fallback
    for k, v in params.items():
        if not hasattr(strategy, k):
            raise AttributeError(f"Unknown strategy parameter: {k!r}")
        setattr(strategy, k, v)
    # Re-validate invariants if dataclass has __post_init__
    post = getattr(strategy, "__post_init__", None)
    if callable(post):
        post()
    return strategy


def _predict_weights_df(strategy: Any, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run strategy inference and extract weights DataFrame.

    Parameters
    ----------
    strategy : Any
        Strategy instance with `predict_weights`.
    df : pd.DataFrame
        Input frame.

    Returns
    -------
    pd.DataFrame
        Weights with CASH column; aligned to df.index.
    """
    res = strategy.predict_weights(df)
    W = res.weights if hasattr(res, "weights") else res  # tolerant
    if not isinstance(W, pd.DataFrame):
        raise TypeError("predict_weights must return a StrategyResult or a DataFrame.")
    if CASH_COL not in W.columns:
        raise KeyError(f"Weights must include CASH column {CASH_COL!r}.")
    return W


def _fit_grid(
    strategy: Any,
    df: pd.DataFrame,
    asset_returns: Union[pd.DataFrame, np.ndarray],
    cfg: LearningConfig,
    opt: OptimizationConfig,
) -> FitResult:
    """
    Grid search over param_grid (exhaustive cartesian product).

    Returns
    -------
    FitResult
        Best params/loss/weights and full history.
    """
    grid = opt.param_grid or {}
    history: List[Tuple[Dict[str, Any], float]] = []
    best_loss = float("inf")
    best_params: Dict[str, Any] = {}
    best_weights: Optional[pd.DataFrame] = None

    # Prepare RF per bar (vector or scalar)
    rf_used = _extract_rf_from_df(df, cfg.rf_per_bar())

    for params in _iter_grid(grid):
        _apply_params(strategy, params)
        W = _predict_weights_df(strategy, df)
        loss_val = _eval_loss_for_weights(W, asset_returns, cfg, rf_per_bar=rf_used)
        history.append((dict(params), loss_val))
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = dict(params)
            best_weights = W

    # If empty grid, evaluate once with current params
    if not grid:
        W = _predict_weights_df(strategy, df)
        loss_val = _eval_loss_for_weights(W, asset_returns, cfg, rf_per_bar=rf_used)
        history.append((dict(), loss_val))
        best_loss = loss_val
        best_params = {}
        best_weights = W

    _apply_params(strategy, best_params)
    assert best_weights is not None
    meta = {"mode": "grid", "evaluations": len(history)}
    return FitResult(best_params, best_loss, best_weights, history, strategy, meta)


def _fit_random(
    strategy: Any,
    df: pd.DataFrame,
    asset_returns: Union[pd.DataFrame, np.ndarray],
    cfg: LearningConfig,
    opt: OptimizationConfig,
) -> FitResult:
    """
    Random search over param distributions.

    Returns
    -------
    FitResult
        Best params/loss/weights and full history.
    """
    dists = opt.param_distributions or {}
    n_iter = int(opt.n_iter) if opt.n_iter is not None else 50

    history: List[Tuple[Dict[str, Any], float]] = []
    best_loss = float("inf")
    best_params: Dict[str, Any] = {}
    best_weights: Optional[pd.DataFrame] = None

    rf_used = _extract_rf_from_df(df, cfg.rf_per_bar())

    for params in _iter_random(dists, n_iter, opt.random_state):
        _apply_params(strategy, params)
        W = _predict_weights_df(strategy, df)
        loss_val = _eval_loss_for_weights(W, asset_returns, cfg, rf_per_bar=rf_used)
        history.append((dict(params), loss_val))
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = dict(params)
            best_weights = W

    # Evaluate current strategy once if no distributions provided
    if not dists:
        W = _predict_weights_df(strategy, df)
        loss_val = _eval_loss_for_weights(W, asset_returns, cfg, rf_per_bar=rf_used)
        history.append((dict(), loss_val))
        best_loss = loss_val
        best_params = {}
        best_weights = W

    _apply_params(strategy, best_params)
    assert best_weights is not None
    meta = {
        "mode": "random",
        "evaluations": len(history),
        "n_iter": n_iter,
        "seed": opt.random_state,
    }
    return FitResult(best_params, best_loss, best_weights, history, strategy, meta)


# ---------------------------------------------------------------------
# Implementation: gradient (torch)
# ---------------------------------------------------------------------


def _fit_gradient(
    strategy: Any,
    df: pd.DataFrame,
    asset_returns: Union[pd.DataFrame, np.ndarray],
    cfg: LearningConfig,
    opt: OptimizationConfig,
) -> FitResult:
    """
    Gradient-based optimization for torch-capable strategies.

    Requirements
    ------------
    Strategy must implement:
      - `forward_torch(df) -> Tensor (T, M)` producing CAUSAL weights (incl. CASH),
      - `torch_parameters() -> Iterable[torch.nn.Parameter]`.

    Returns
    -------
    FitResult
        Best params/loss/weights and training meta.

    Raises
    ------
    RuntimeError
        If PyTorch is not available or the strategy lacks the required methods.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not available; use 'grid' or 'random' mode.")

    fwd = getattr(strategy, "forward_torch", None)
    get_params = getattr(strategy, "torch_parameters", None)
    if not callable(fwd) or not callable(get_params):
        raise RuntimeError(
            "Strategy must implement forward_torch() and torch_parameters()."
        )

    device = _torch_device(opt)
    assert device is not None
    params = list(get_params())
    if len(params) == 0:
        raise RuntimeError("No trainable torch parameters found.")

    # Prepare returns as torch Tensor on the same device/dtype
    R_np = _ensure_returns(asset_returns)
    R = torch.as_tensor(R_np, dtype=torch.float32, device=device)

    # RF per bar (vector or scalar), and Objective with annualizer set
    rf_np = _extract_rf_from_df(df, cfg.rf_per_bar())
    if isinstance(rf_np, np.ndarray):
        rf_t = torch.as_tensor(rf_np, dtype=torch.float32, device=device)
    else:
        rf_t = torch.as_tensor(float(rf_np), dtype=torch.float32, device=device)

    # If cfg.bars_per_unit <= 0, infer from df (already handled in fit_strategy)
    obj = _objective_from_config(cfg)
    if obj.name.lower() == "neg_sharpe":
        obj.rf_per_period = rf_t  # tensor ok; losses._as_array will handle

    # Choose optimizer
    opt_name = str(opt.optimizer).lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            params, lr=float(opt.lr), weight_decay=float(opt.weight_decay)
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=float(opt.lr), weight_decay=float(opt.weight_decay)
        )
    elif opt_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            params, lr=float(opt.lr), history_size=10, max_iter=20
        )
    else:
        raise ValueError("optimizer must be one of {'adam','sgd','lbfgs'}")

    best_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    history: List[Tuple[Dict[str, Any], float]] = []

    # Closure for LBFGS
    def closure() -> TorchTensor:  # type: ignore[name-defined]
        optimizer.zero_grad(set_to_none=True)
        W = fwd(df)  # expected (T, M) torch Tensor with CASH column
        if not isinstance(W, TorchTensor):  # type: ignore[name-defined]
            raise RuntimeError("forward_torch(df) must return a torch.Tensor.")
        # Evaluate objective directly on tensors
        loss_tensor = evaluate_objective(
            weights=W,
            asset_returns=R,
            rf=rf_t,
            has_cash_in_returns=cfg.has_cash_in_returns,
            cash_col_idx=cfg.cash_col_idx,
            cost_items=None,
            cost_config=cfg.cost_config,
            fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps,
            per_asset_multipliers=cfg.per_asset_multipliers,
            initial_weights=cfg.initial_weights,
            spec=obj,
        )
        # ensure tensor scalar
        if not hasattr(loss_tensor, "backward"):
            loss_tensor = torch.as_tensor(
                loss_tensor, dtype=torch.float32, device=device
            )
        loss_tensor.backward()
        return loss_tensor

    epochs = int(opt.epochs)
    for epoch in range(max(1, epochs)):
        if opt_name == "lbfgs":
            loss_t = closure()
            optimizer.step(closure)  # LBFGS calls closure internally
        else:
            loss_t = closure()
            optimizer.step()
        loss_val = float(loss_t.detach().cpu().item())
        history.append(
            ({}, loss_val)
        )  # params are inside torch; we don't enumerate here

        if loss_val < best_loss:
            best_loss = loss_val
            # Save best state dict snapshot
            best_state = {
                k: v.clone().detach().cpu()
                for k, v in _state_dict_from_params(params).items()
            }

    # Restore best params if improved
    if best_state is not None:
        _load_state_into_params(params, best_state)

    # Produce final best weights as DataFrame
    with torch.no_grad():
        W_t = fwd(df)
    if not isinstance(W_t, TorchTensor):  # type: ignore[name-defined]
        raise RuntimeError("forward_torch(df) must return a torch.Tensor.")
    W_np = W_t.detach().cpu().numpy()
    if not isinstance(df.index, pd.DatetimeIndex):
        idx = pd.RangeIndex(W_np.shape[0])
    else:
        idx = df.index
    # Column names: try to use strategy-provided risky names + CASH
    if hasattr(strategy, "asset"):
        cols = [getattr(strategy, "asset"), CASH_COL]
        if W_np.shape[1] != 2:
            cols = [f"asset_{j}" for j in range(W_np.shape[1] - 1)] + [CASH_COL]
    elif hasattr(strategy, "assets"):
        risky = list(getattr(strategy, "assets"))
        cols = risky + [CASH_COL]
        if len(cols) != W_np.shape[1]:
            cols = [f"asset_{j}" for j in range(W_np.shape[1] - 1)] + [CASH_COL]
    else:
        cols = [f"asset_{j}" for j in range(W_np.shape[1] - 1)] + [CASH_COL]
    W_df = pd.DataFrame(W_np, index=idx, columns=cols)

    meta = {
        "mode": "gradient",
        "optimizer": opt_name,
        "epochs": epochs,
        "device": str(device),
    }
    return FitResult(
        best_params={},
        best_loss=best_loss,
        best_weights=W_df,
        history=history,
        updated_strategy=strategy,
        meta=meta,
    )


def _state_dict_from_params(params: Sequence[Any]) -> Dict[str, torch.Tensor]:  # type: ignore[name-defined]
    """
    Build a flat state dict from a list of torch parameters.

    Parameters
    ----------
    params : Sequence
        Iterable of torch.nn.Parameter.

    Returns
    -------
    Dict[str, torch.Tensor]
        Mapping name_i → tensor value.
    """
    state: Dict[str, torch.Tensor] = {}
    for i, p in enumerate(params):
        state[f"p_{i}"] = p.data.detach().clone()
    return state


def _load_state_into_params(params: Sequence[Any], state: Mapping[str, torch.Tensor]) -> None:  # type: ignore[name-defined]
    """
    Load tensors back into a list of torch parameters.

    Parameters
    ----------
    params : Sequence
        Iterable of torch.nn.Parameter.
    state : Mapping[str, torch.Tensor]
        Flat mapping produced by `_state_dict_from_params`.
    """
    for i, p in enumerate(params):
        key = f"p_{i}"
        if key in state:
            p.data.copy_(state[key].to(p.data.device))


def _mk_learning(**kwargs) -> LearningConfig:
    allowed = {f.name for f in fields(LearningConfig)}
    payload = {k: v for k, v in kwargs.items() if k in allowed}
    print("[learning] applying keys:", sorted(payload.keys()))
    return LearningConfig(**payload)


def _mk_optimization(**kwargs) -> OptimizationConfig:
    allowed = {f.name for f in fields(OptimizationConfig)}
    payload = {k: v for k, v in kwargs.items() if k in allowed}
    print("[optimization] applying keys:", sorted(payload.keys()))
    return OptimizationConfig(**payload)

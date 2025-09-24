# strategies.py
"""
Strategy interfaces that produce portfolio WEIGHTS

Design
------
- Each strategy consumes a DataFrame (single- or multi-asset) and outputs a
  time-indexed DataFrame of *weights* that sum to 1.0 on each row, including a
  CASH column. The weights are CAUSAL: decisions at time t are intended to be
  applied starting at t + trade_lag_bars (default 1).
- Parameters are stored on the dataclass and can be updated later via set_params.
- Extra diagnostic series (features, raw signals, thresholds) are returned via
  StrategyResult.artifacts.

Conventions
-----------
- Single-asset df columns: ["open","high","low","close","volume", ...features...]
- Multi-asset df columns: MultiIndex (asset, field), e.g., ("BTCUSDT","close")
- Weights columns:
    * Single-asset: [<asset_symbol>, CASH]
    * Multi-asset:  [asset_1, asset_2, ..., CASH]
- Index: tz-aware increasing DatetimeIndex (UTC recommended)
"""


from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Mapping, Optional, Tuple, List, Union

import numpy as np
import pandas as pd

__all__ = [
    "StrategyResult",
    "BaseStrategy",
    "SingleAssetMACrossover",
    # helpers (optional export if you want to use them externally)
    "_validate_index",
    "_to_df",
    "_clip_weights_rowwise",
    "_normalize_simplex_with_cash",
    "_apply_rebalance_every",
    "_apply_causal_lag",
    "_enforce_row_sums_one",
]

try:
    from hedge.portfolio import CASH_COL  # expected constant
except Exception:
    CASH_COL = "CASH"

from hedge.utils import ensure_series
from hedge.features import sma


# ==============================================================================
# Results
# ==============================================================================


@dataclass(frozen=True)
class StrategyResult:
    """
    Output of a strategy run.

    Attributes
    ----------
    weights : pd.DataFrame
        Row-wise portfolio weights (sum to 1.0), including CASH. Index = df.index.
    artifacts : Dict[str, Any]
        Optional extras (features, raw signals, debug frames). Keep JSON-friendly.
    meta : Dict[str, Any]
        Small metadata dict (parameters used, data shape, etc.).
    """

    weights: pd.DataFrame
    artifacts: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Generic helpers
# ==============================================================================


def _validate_index(idx: pd.Index, *, require_tz: bool = True) -> None:
    """
    Validate index is a monotonic increasing DatetimeIndex (tz-aware if required).

    Parameters
    ----------
    idx : pd.Index
        Index to validate.
    require_tz : bool, optional
        If True, enforce tz-aware DatetimeIndex. Default True.

    Raises
    ------
    TypeError
        If index is not a DatetimeIndex (when required).
    ValueError
        If index is not monotonic increasing or tz-awareness is missing.
    """
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")
    if not idx.is_monotonic_increasing:
        raise ValueError("Index must be monotonic increasing.")
    if require_tz and idx.tz is None:
        raise ValueError("Index must be tz-aware (UTC recommended).")


def _to_df(
    obj: Union[pd.Series, pd.DataFrame], name: Optional[str] = None
) -> pd.DataFrame:
    """
    Coerce Series to DataFrame (with provided column name) or keep DataFrame.

    Parameters
    ----------
    obj : pd.Series or pd.DataFrame
        Input time series or frame.
    name : str, optional
        Column name if `obj` is a Series.

    Returns
    -------
    pd.DataFrame
        DataFrame with a single column if input was a Series.
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if name is None:
        name = obj.name or "value"
    return obj.to_frame(name=name)


def _clip_weights_rowwise(W: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """
    Clip each row elementwise into [lo, hi] without renormalization.

    Parameters
    ----------
    W : pd.DataFrame
        Risky weights frame.
    lo : float
        Lower bound per element.
    hi : float
        Upper bound per element.

    Returns
    -------
    pd.DataFrame
        Clipped weights (same shape).
    """
    return W.clip(lower=lo, upper=hi)


def _normalize_simplex_with_cash(
    W_risky: pd.DataFrame, cash_col: str = CASH_COL
) -> pd.DataFrame:
    """
    Build a full weight matrix summing to 1.0 by adding CASH as residual.

    Rules
    -----
    - If a row has all-NaN risky weights, allocate 100% to CASH.
    - Otherwise: risky_sum = sum(risky_i), cash = 1 - risky_sum.
      (No explicit leverage control here; gross may exceed 1.0 if shorts are used.)

    Parameters
    ----------
    W_risky : pd.DataFrame
        Risky-leg weights (may contain NaNs).
    cash_col : str, optional
        CASH column name.

    Returns
    -------
    pd.DataFrame
        Columns = risky_cols + [cash_col], per-row sum = 1.0.
    """
    W = W_risky.copy()
    risky_cols = list(W.columns)
    s = W.sum(axis=1)
    cash = 1.0 - s

    out = W.copy()
    out[cash_col] = cash

    # All-NaN risky → 100% CASH
    all_nan_mask = W.isna().all(axis=1)
    out.loc[all_nan_mask, risky_cols] = 0.0
    out.loc[all_nan_mask, cash_col] = 1.0

    # Defensive: isolated NaNs to 0.0 (row-sum stays 1.0)
    out = out.fillna(0.0)
    return out


def _apply_rebalance_every(W: pd.DataFrame, every: Optional[int]) -> pd.DataFrame:
    """
    Apply a bar-based rebalance schedule by holding weights between rebalance bars.

    Parameters
    ----------
    W : pd.DataFrame
        Row-normalized weights (include CASH).
    every : int or None
        Rebalance every N bars. If None or <=1, returns W unchanged.

    Returns
    -------
    pd.DataFrame
        Weights piecewise constant between rebalance bars.
    """
    if every is None or every <= 1:
        return W
    n = len(W)
    mask = np.zeros(n, dtype=bool)
    mask[:: int(every)] = True  # rebalance on these rows
    W_sched = W.copy()
    W_sched.loc[~pd.Series(mask, index=W.index)] = np.nan
    return W_sched.ffill()


def _apply_causal_lag(
    W: pd.DataFrame, lag: int, cash_col: str = CASH_COL
) -> pd.DataFrame:
    """
    Shift weights forward by `lag` bars so that decisions at t apply from t+lag.

    - Typical choice: lag=1 (decide at close t, apply at next bar).
    - Leading rows introduced by shift are filled with a neutral allocation (all CASH).

    Parameters
    ----------
    W : pd.DataFrame
        Weight matrix including CASH.
    lag : int
        Bars to shift forward (causality).
    cash_col : str, optional
        CASH column name.

    Returns
    -------
    pd.DataFrame
        Lagged weights with neutral fill on leading rows.
    """
    if lag <= 0:
        return W
    W_lag = W.shift(lag)
    neutral = pd.DataFrame(0.0, index=W.index, columns=W.columns, dtype=float)
    if cash_col in W.columns:
        neutral[cash_col] = 1.0
    return W_lag.combine_first(neutral)


def _enforce_row_sums_one(
    W: pd.DataFrame,
    *,
    cash_col: str = CASH_COL,
    atol: float = 1e-9,
) -> pd.DataFrame:
    """
    Finalize weights so each row sums to 1.0 without renormalizing non-NaN risky legs.

    Rules
    -----
    - If an entire row is NaN: set all risky=0, CASH=1.
    - NaNs in risky columns → 0 (do not touch non-NaN values).
    - CASH is recomputed as residual: 1 - sum(risky).
    - Validate totals within tolerance.

    Parameters
    ----------
    W : pd.DataFrame
        Allocation matrix; must include `cash_col`.
    cash_col : str, optional
        Name of the cash column.
    atol : float, optional
        Absolute tolerance for validation.

    Returns
    -------
    pd.DataFrame
        Same shape; per-row sum == 1 within tolerance.
    """
    if cash_col not in W.columns:
        raise KeyError(f"Missing required cash column: {cash_col!r}")

    out = W.copy()
    risky_cols = [c for c in out.columns if c != cash_col]

    all_nan = out.isna().all(axis=1)
    if all_nan.any():
        out.loc[all_nan, :] = 0.0
        out.loc[all_nan, cash_col] = 1.0

    if risky_cols:
        out.loc[:, risky_cols] = out.loc[:, risky_cols].fillna(0.0)

    risky_sum = (
        out[risky_cols].sum(axis=1) if risky_cols else pd.Series(0.0, index=out.index)
    )
    out[cash_col] = 1.0 - risky_sum

    out = out.mask(out.abs() < atol, 0.0)
    total_err = (out.sum(axis=1) - 1.0).abs()
    if (total_err > atol).any():
        bad = out.loc[total_err > atol].head(3)
        raise ValueError(
            "Row sums deviate from 1.0 beyond tolerance after finalization.\n"
            f"Examples (up to 3 rows):\n{bad}"
        )
    return out


# ==============================================================================
# Base Strategy
# ==============================================================================


@dataclass
class BaseStrategy:
    """
    Abstract base for all strategies that produce WEIGHTS (omega_t).

    Parameters
    ----------
    name : str
        Identifier for reporting.
    version : str
        Semantic version of the implementation/spec.
    random_state : Optional[int]
        RNG seed (for ML/agents later).
    warmup_bars : int
        Bars to neutralize at the start (insufficient history).
    trade_lag_bars : int
        Causality: shift decisions forward by this many bars (default 1).
    rebalance_every : Optional[int]
        If provided (>1), hold weights between rebalance bars and update only
        every N bars (applied before causal lag).

    Notes
    -----
    Subclasses must implement:
      - is_multi_asset()
      - predict_weights(df) -> StrategyResult
    """

    name: str
    version: str = "0.1.0"
    random_state: Optional[int] = None
    warmup_bars: int = 0
    trade_lag_bars: int = 1
    rebalance_every: Optional[int] = None  # bars; None → every bar

    # ---------- core inference ----------

    def predict_weights(self, df: pd.DataFrame) -> StrategyResult:
        """
        Compute a causal, row-normalized weight matrix (includes CASH).

        Parameters
        ----------
        df : pd.DataFrame
            Strategy input frame.

        Returns
        -------
        StrategyResult
            weights: DataFrame with row-sum == 1.0 (includes CASH), aligned to df.index.
        """
        raise NotImplementedError

    # ---------- scope ----------

    def is_multi_asset(self) -> bool:
        """
        Whether the strategy expects multi-asset (column MultiIndex) input.

        Returns
        -------
        bool
            True for multi-asset, False otherwise.
        """
        raise NotImplementedError

    def requires_features(self) -> List[str]:
        """
        List of feature names required by the strategy (for pipeline orchestration).

        Returns
        -------
        List[str]
            Names of required features. Empty if the strategy computes them internally.
        """
        return []

    # ---------- params I/O ----------

    def get_params(self) -> Dict[str, Any]:
        """
        Export strategy parameters as a plain dict.

        Returns
        -------
        Dict[str, Any]
            Dataclass fields as a dictionary.
        """
        return asdict(self)

    def set_params(self, **params: Any) -> "BaseStrategy":
        """
        Update strategy parameters in-place.

        Parameters
        ----------
        **params : Any
            Field values to set.

        Returns
        -------
        BaseStrategy
            Self (with updated fields).
        """
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown parameter: {k!r}")
            setattr(self, k, v)
        # Re-validate invariants via __post_init__ if present
        post = getattr(self, "__post_init__", None)
        if callable(post):
            post()
        return self

    # ---------- persistence ----------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize strategy configuration.

        Returns
        -------
        Dict[str, Any]
            Dataclass fields as a dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaseStrategy":
        """
        Restore strategy from a dict.

        Parameters
        ----------
        payload : Mapping[str, Any]
            Serialized fields.

        Returns
        -------
        BaseStrategy
            New instance.
        """
        return cls(**dict(payload))


# ==============================================================================
# Single-asset baseline: SMA crossover
# ==============================================================================


@dataclass
class SingleAssetMACrossover(BaseStrategy):
    """
    Single-asset SMA(fast) vs SMA(slow) → portfolio weights.

    Parameters
    ----------
    asset : str
        Symbol used for the risky column in the weight matrix.
    price_field : str
        Column in df to use as price (e.g., "close").
    fast_window : int
        Lookback for fast SMA.
    slow_window : int
        Lookback for slow SMA.
    w_long : float
        Risk allocation to the asset when fast > slow (e.g., 1.0 → fully invested).
    w_short : float
        Risk allocation when fast < slow (e.g., -1.0 → short if allowed).
        If your environment doesn’t allow shorting, set w_short = 0.0.
    clip_per_asset : float
        Per-row clipping bound for the risky asset weight before adding CASH.

    Notes
    -----
    - This baseline computes SMAs internally; `requires_features()` is empty.
    - No explicit leverage control: CASH is residual (1 - sum(risky)).
    """

    name: str = "single-asset-ma-crossover"
    asset: str = "BTCUSDT"
    price_field: str = "close"
    fast_window: int = 20
    slow_window: int = 50
    w_long: float = 1.0
    w_short: float = 0.0  # default: no shorting
    clip_per_asset: float = 1.0

    def __post_init__(self):
        if self.slow_window < 1 or self.fast_window < 1:
            raise ValueError("windows must be >= 1")
        if self.slow_window < self.fast_window:
            raise ValueError("Slow window width must exceed fast window width")
        self.warmup_bars = max(self.warmup_bars, self.slow_window)

    def is_multi_asset(self) -> bool:
        """
        Single-asset strategy indicator.

        Returns
        -------
        bool
            Always False for this strategy.
        """
        return False

    def predict_weights(self, df: pd.DataFrame) -> StrategyResult:
        """
        Compute causal weights on [asset, CASH] via SMA crossover.

        Parameters
        ----------
        df : pd.DataFrame
            Input data frame with at least `price_field` column.

        Returns
        -------
        StrategyResult
            Weights including CASH; row-sum = 1.0; aligned to df.index.
        """
        _validate_index(df.index, require_tz=True)

        if self.price_field not in df.columns:
            raise KeyError(f"Missing price field '{self.price_field}'.")
        px = ensure_series(df[self.price_field], name=self.price_field).astype(float)

        fast = sma(px, self.fast_window)
        slow = sma(px, self.slow_window)

        diff = (fast - slow).astype(float)
        diff = diff.where(diff.notna(), other=0.0)
        sig = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
        sig = pd.Series(sig, index=px.index, name="signal").astype(float)

        w_asset = np.where(sig > 0, self.w_long, np.where(sig < 0, self.w_short, 0.0))
        w_asset = pd.Series(w_asset, index=px.index, name=self.asset).astype(float)

        warm = int(self.warmup_bars)
        if warm > 0 and len(w_asset) > 0:
            w_asset.iloc[:warm] = 0.0

        W_risky = _to_df(w_asset, name=self.asset)
        W_risky = _clip_weights_rowwise(
            W_risky, -self.clip_per_asset, self.clip_per_asset
        )

        W_full = _normalize_simplex_with_cash(W_risky, cash_col=CASH_COL)

        W_full = _apply_rebalance_every(W_full, self.rebalance_every)

        W_full = _apply_causal_lag(W_full, lag=self.trade_lag_bars, cash_col=CASH_COL)

        W_full = _enforce_row_sums_one(W_full, cash_col=CASH_COL)

        artifacts: Dict[str, Any] = {
            "sma_fast": fast.rename(f"sma_{self.fast_window}"),
            "sma_slow": slow.rename(f"sma_{self.slow_window}"),
            "raw_signal": sig,
            "raw_weight_asset": w_asset,
        }
        meta = {
            "asset": self.asset,
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "w_long": self.w_long,
            "w_short": self.w_short,
            "warmup_bars": self.warmup_bars,
            "trade_lag_bars": self.trade_lag_bars,
            "rebalance_every": self.rebalance_every,
            "cash_col": CASH_COL,
        }
        return StrategyResult(weights=W_full, artifacts=artifacts, meta=meta)

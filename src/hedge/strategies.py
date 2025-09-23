# strategies.py
"""
Strategy interfaces that produce portfolio WEIGHTS (omega_t), plus helpers.

Design
------
- Each strategy consumes a DataFrame (single- or multi-asset) and outputs a
  time-indexed DataFrame of *weights* that sum to 1.0 on each row, including a
  'CASH' column. The weights are CAUSAL: decisions at time t are intended to be
  applied starting at t + trade_lag_bars (default 1).
- Parameters are stored on the dataclass and can be updated later (fit/update).
- Extra diagnostic series (features, raw signals, thresholds) are returned via
  StrategyResult.artifacts.

Conventions
-----------
- Single-asset df columns: ["open","high","low","close","volume", ...features...]
- Multi-asset df columns: MultiIndex (asset, field), e.g., ("BTCUSDT","close")
- Weights columns:
    * Single-asset: [<asset_symbol>, "CASH"]
    * Multi-asset:  [asset_1, asset_2, ..., "CASH"]
- Index: tz-aware increasing DatetimeIndex
"""


from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Mapping, Optional, Tuple, List, Union

import numpy as np
import pandas as pd

from hedge.utils import ensure_series  # your existing helper
from hedge.features import sma  # used by the baseline strategy

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
        Row-wise portfolio weights (sum to 1.0), including 'CASH'. Index = df.index.
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


def _to_df(
    obj: Union[pd.Series, pd.DataFrame], name: Optional[str] = None
) -> pd.DataFrame:
    """Coerce Series→DataFrame (with provided column name) or keep DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if name is None:
        name = obj.name or "value"
    return obj.to_frame(name=name)


def _clip_weights_rowwise(W: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """
    Clip each row elementwise into [lo, hi]. Does NOT renormalize.
    """
    return W.clip(lower=lo, upper=hi)


def _normalize_simplex_with_cash(
    W_risky: pd.DataFrame, cash_col: str = "CASH"
) -> pd.DataFrame:
    """
    Given risky weights (may contain NaNs), build a full weight matrix that sums to 1.

    Rules
    -----
    - If a row has all-NaN risky weights, allocate 100% to CASH.
    - Otherwise:
        risky_sum = sum(risky_i), cash = 1 - risky_sum
      (No leverage here)

    Returns
    -------
    pd.DataFrame with the same index, columns = risky_cols + [cash_col], row-sum = 1.
    """
    W = W_risky.copy()
    risky_cols = list(W.columns)
    s = W.sum(axis=1)
    cash = 1.0 - s

    out = W.copy()
    out[cash_col] = cash

    # Where all risky are NaN → 100% CASH
    all_nan_mask = W.isna().all(axis=1)
    out.loc[all_nan_mask, risky_cols] = 0.0
    out.loc[all_nan_mask, cash_col] = 1.0

    # Fill any isolated NaNs with 0.0 (defensive); then row-sum remains exactly 1.
    out = out.fillna(0.0)
    return out


def _apply_causal_lag(W: pd.DataFrame, lag: int) -> pd.DataFrame:
    """
    Shift weights forward by `lag` bars so that decisions at t are applied from t+lag.

    - Typical choice: lag=1 (decide at close t, apply at next bar).
    - Leading rows introduced by shift are NaN; we fill them with a neutral allocation (all CASH).
    """
    if lag <= 0:
        return W
    W_lag = W.shift(lag)
    if "CASH" in W.columns:
        neutral = pd.DataFrame(0.0, index=W.index, columns=W.columns, dtype=float)
        neutral["CASH"] = 1.0
    else:
        # Fallback: if no explicit CASH column yet, fill with zeros (caller may add cash later)
        neutral = pd.DataFrame(0.0, index=W.index, columns=W.columns, dtype=float)
    return W_lag.combine_first(neutral)


def _enforce_row_sums_one(
    W: pd.DataFrame,
    *,
    cash_col: str = "CASH",
    atol: float = 1e-9,
) -> pd.DataFrame:
    """
    Minimal post-lag fixups so each row sums to 1.0 without re-normalizing
    risky legs. Only handles edge-cases introduced by shifting/alignment.

    Rules
    -----
    - If an entire row is NaN: set all risky=0, CASH=1.
    - NaNs in risky columns -> 0 (do not touch non-NaN values).
    - CASH is recomputed as residual: 1 - sum(risky).
    - Validate totals within tolerance

    Parameters
    ----------
    W : pd.DataFrame
        Allocation matrix; must include `cash_col`.
    cash_col : str, default "CASH"
        Name of the cash column.
    atol : float, default 1e-9
        Absolute tolerance for numerical cleanups and validation.

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

    # Recompute CASH as residual
    risky_sum = (
        out[risky_cols].sum(axis=1) if risky_cols else pd.Series(0.0, index=out.index)
    )
    out[cash_col] = 1.0 - risky_sum

    out = out.mask(out.abs() < atol, 0.0)
    out[cash_col] = out[cash_col].mask(
        (out[cash_col] > 1.0 - atol) & (out[cash_col] <= 1.0 + atol), 1.0
    )

    # Validate totals
    total_err = (out.sum(axis=1) - 1.0).abs()
    if (total_err > atol).any():
        bad = out.loc[total_err > atol].head(3)
        raise ValueError(
            "Row sums deviate from 1.0 beyond tolerance after minimal fixes.\n"
            f"Examples (up to 3 rows):\n{bad}"
        )

    return out


# ==============================================================================
# Base Strategy (weights-producing)
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

    # ---------- lifecycle ----------

    def fit(self, df: pd.DataFrame) -> "BaseStrategy":
        """Optional training/calibration. Default: no-op."""
        return self

    def update(self, df_new: pd.DataFrame) -> "BaseStrategy":
        """Optional online update for adaptive/AI agents."""
        return self

    # ---------- core inference ----------

    def predict_weights(self, df: pd.DataFrame) -> StrategyResult:
        """
        Compute a causal, row-normalized weight matrix (includes 'CASH').

        Returns
        -------
        StrategyResult
            weights: DataFrame with row-sum == 1.0 (includes CASH), aligned to df.index.
        """
        raise NotImplementedError

    # ---------- scope ----------

    def is_multi_asset(self) -> bool:
        """Return True if the strategy expects multi-asset (column MultiIndex) input."""
        raise NotImplementedError

    # ---------- persistence ----------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaseStrategy":
        return cls(**dict(payload))


# ==============================================================================
# Single-asset baseline: SMA crossover → weights on [ASSET, CASH]
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
        return False

    def fit(self, df: pd.DataFrame) -> "SingleAssetMACrossover":
        """
        Placeholder
        """
        return self

    def predict_weights(self, df: pd.DataFrame) -> StrategyResult:

        if self.price_field not in df.columns:
            raise KeyError(f"Missing price field '{self.price_field}'.")
        px = ensure_series(df[self.price_field], name=self.price_field).astype(float)

        fast = sma(px, self.fast_window)
        slow = sma(px, self.slow_window)

        # Raw, discrete regime signal {-1,0,+1}
        diff = fast - slow
        sig = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
        sig = pd.Series(sig, index=px.index, name="signal").astype(float)

        # Map signal → risky weight for the asset (before CASH)
        w_asset = np.where(sig > 0, self.w_long, np.where(sig < 0, self.w_short, 0.0))
        w_asset = pd.Series(w_asset, index=px.index, name=self.asset).astype(float)

        warm = int(self.warmup_bars)
        if warm > 0 and len(w_asset) > 0:
            w_asset.iloc[:warm] = 0.0

        #   Clip per-asset if needed (no leverage here)
        W_risky = _to_df(w_asset, name=self.asset)
        W_risky = _clip_weights_rowwise(
            W_risky, -self.clip_per_asset, self.clip_per_asset
        )

        W_full = _normalize_simplex_with_cash(W_risky, cash_col="CASH")

        W_full = _apply_causal_lag(W_full, lag=self.trade_lag_bars)

        W_full = _enforce_row_sums_one(W_full)

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
        }
        return StrategyResult(weights=W_full, artifacts=artifacts, meta=meta)


@dataclass
class CrossSectionalMomentum(BaseStrategy):
    """
    Multi-asset cross-sectional momentum → weights.

    Parameters
    ----------
    field : str
        Per-asset column (e.g., 'close') in a MultiIndex df (asset, field).
    lookback : int
        Return lookback for ranking.
    top_k : Optional[int]
        How many assets to allocate to (equal weight among selected); rest to CASH.
        If None, allocate proportionally to positive ranks later.
    clip_per_asset : float
        Per-asset clip before cash residual.

    Notes
    -----
    - Implement predict_weights() later.
    - Output columns: [asset_1, ..., asset_n, "CASH"].
    """

    name: str = "xsec-momentum"
    field: str = "close"
    lookback: int = 20
    top_k: Optional[int] = None
    clip_per_asset: float = 1.0

    def is_multi_asset(self) -> bool:
        return True

    def predict_weights(self, df: pd.DataFrame) -> StrategyResult:
        raise NotImplementedError("Implement cross-sectional momentum later.")


@dataclass
class EnsembleWeights(BaseStrategy):
    """
    Weighted ensemble of weights-producing strategies.

    Parameters
    ----------
    members : Tuple[BaseStrategy, ...]
        Child strategies (already configured/fitted as needed).
    weights : Optional[Tuple[float, ...]]
        Linear combination weights for member outputs; default equal.

    Notes
    -----
    - Implement alignment & mixing later; enforce row-sum 1.0 and include CASH.
    """

    name: str = "ensemble-weights"
    members: Tuple[BaseStrategy, ...] = field(default_factory=tuple)
    weights: Optional[Tuple[float, ...]] = None

    def is_multi_asset(self) -> bool:
        # If any member is multi-asset, treat ensemble as multi-asset.
        return any(m.is_multi_asset() for m in self.members)

    def predict_weights(self, df: pd.DataFrame) -> StrategyResult:
        raise NotImplementedError("Implement ensemble mixing later.")

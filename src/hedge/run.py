# run.py
"""
High-level runners for training and evaluating strategies on CSV data.

This module exposes two workflows:

1) train_on_csv(...)
   - Load an OHLCV or portfolio CSV.
   - Initialize a strategy by name (from a registry) with given params.
   - Prepare data per strategy scope (single-asset vs multi-asset).
   - Fit with a chosen optimization mode/objective (via learning.py).
   - Save the fitted model + training metadata into ./models.
   - Return (strategy, FitResult, model_path).

2) run_on_csv(...)
   - Load an OHLCV (or portfolio) CSV.
   - Load a strategy (object or saved JSON path) or accept an instance.
   - Generate weights, convert to holdings, compute requested metrics.
   - Return a RunResult with weights, holdings, equity, trades, metrics.

Notes
-----
- Portfolio CSVs are expected to have MultiIndex columns (symbol, field), where
  the field includes at least "close" (and optionally open/high/low/volume).
- Single-asset CSVs must have columns: timestamp, open, high, low, close, volume.
- Risk-free 'rf' is optional; if provided as a per-bar Series aligned to the
  index, it is used in holdings conversion and Sharpe (see metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union, List
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Project imports
# make load_portfolio_csv optional to avoid hard dependency if not exported
try:
    from hedge.data import DataScheme, load_ohlcv_csv, load_portfolio_csv  # type: ignore
except Exception:  # pragma: no cover
    from hedge.data import DataScheme, load_ohlcv_csv  # type: ignore

    load_portfolio_csv = None  # type: ignore[assignment]

from hedge.strategies import (
    BaseStrategy,
    SingleAssetMACrossover,
    StrategyResult,
)
from hedge.fitting.learning import (
    LearningConfig,
    OptimizationConfig,
    FitResult as FitResultLearning,
    fit_strategy,
)
from hedge.portfolio import (
    CASH_COL,
    portfolio_returns,
    weights_to_holdings_df,
    HoldingsResult,
)
from hedge.metrics import (
    equity_curve,
    roi,
    cagr,
    sharpe,
    max_drawdown,
    calmar,
    hit_rate,
)

# ---------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, Any] = {
    "single-asset-ma-crossover": SingleAssetMACrossover,
    # "xsec-momentum": CrossSectionalMomentum,  # when implemented
    # "ensemble-weights": EnsembleWeights,      # when implemented
}


def _resolve_strategy(
    name_or_obj: Union[str, BaseStrategy], **params: Any
) -> BaseStrategy:
    """
    Resolve a strategy: if an instance is passed, return it; otherwise construct by name.
    """
    if isinstance(name_or_obj, BaseStrategy):
        return name_or_obj
    name = str(name_or_obj).lower()
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY.keys())}"
        )
    cls = STRATEGY_REGISTRY[name]
    return cls(**params)


# ---------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _model_payload(
    strategy: BaseStrategy,
    fit: Optional[FitResultLearning],
    data_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "saved_at_utc": _utc_now_iso(),
        "strategy_class": strategy.__class__.__name__,
        "strategy_module": strategy.__class__.__module__,
        "strategy_name": getattr(strategy, "name", strategy.__class__.__name__).lower(),
        "strategy_version": getattr(strategy, "version", "0.0.0"),
        "strategy_config": (
            strategy.to_dict() if hasattr(strategy, "to_dict") else asdict(strategy)
        ),
        "data_meta": dict(data_meta),
    }
    if fit is not None:
        payload["fit"] = {
            "best_params": fit.best_params,
            "best_loss": fit.best_loss,
            "meta": fit.meta,
            "history": fit.history,
        }
    return payload


# --- add to run.py (helpers section) ---


def _coerce_rf_like(rf_obj: Any, T: int) -> Any:
    """
    Normalize risk-free input so portfolio alignment can always assign into a 1-D slice.

    Accepts:
      - scalar (float/int) -> float
      - pd.Series of length T -> 1-D np.ndarray (T,)
      - pd.DataFrame with one column (T,1) -> 1-D np.ndarray (T,)
      - np.ndarray (T,) -> 1-D unchanged
      - np.ndarray (T,1) -> squeezed to (T,)

    Raises if given a mismatched length.
    """
    import numpy as _np
    import pandas as _pd

    if rf_obj is None:
        return 0.0
    # Scalars
    if _np.isscalar(rf_obj):
        return float(rf_obj)

    # pandas
    if isinstance(rf_obj, _pd.Series):
        arr = rf_obj.to_numpy()
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.shape[0] != T:
            raise ValueError(f"rf Series length {arr.shape[0]} != T={T}")
        return arr

    if isinstance(rf_obj, _pd.DataFrame):
        if rf_obj.shape[1] != 1:
            raise ValueError("rf DataFrame must have exactly one column.")
        arr = rf_obj.to_numpy().reshape(-1)  # (T,1) -> (T,)
        if arr.shape[0] != T:
            raise ValueError(f"rf DataFrame length {arr.shape[0]} != T={T}")
        return arr

    # numpy
    arr = _np.asarray(rf_obj)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    elif arr.ndim > 1:
        raise ValueError("rf must be scalar, (T,), or (T,1).")
    if arr.ndim == 1 and arr.shape[0] != T:
        raise ValueError(f"rf length {arr.shape[0]} != T={T}")
    return arr


def _save_model_json(
    payload: Mapping[str, Any],
    out_dir: Union[str, Path],
    code_hint: Optional[str] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = payload.get("strategy_name", "strategy")
    code = code_hint or payload.get("data_meta", {}).get("code", None) or "data"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = f"{name}_{code}_{ts}.json"
    path = out_dir / fname
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_model(path: Union[str, Path]) -> BaseStrategy:
    """
    Load a saved model JSON and reconstruct the strategy instance.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cls_name = payload["strategy_class"]
    module_name = payload["strategy_module"]
    # Try registry first
    for _, cls in STRATEGY_REGISTRY.items():
        if cls.__name__ == cls_name:
            return cls.from_dict(payload["strategy_config"])
    # Fallback: dynamic import
    mod = __import__(module_name, fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    return cls.from_dict(payload["strategy_config"])


# ---------------------------------------------------------------------
# CSV loaders & training helpers
# ---------------------------------------------------------------------


def _generic_load_portfolio_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Robustly load a 'portfolio_<code>.csv' saved with MultiIndex columns
    (two header rows) OR a flat form like 'BTCUSDT.close'.

    Returns a DataFrame with:
      - tz-aware UTC DatetimeIndex named 'timestamp'
      - columns either MultiIndex (symbol, field) or flat per-asset
    """
    p = Path(path)
    # First try two-row header (what pandas writes for MultiIndex columns)
    try:
        df = pd.read_csv(p, header=[0, 1])
        if isinstance(df.columns, pd.MultiIndex):
            # find the timestamp column at level 0
            ts_candidates = [
                c for c in df.columns if isinstance(c, tuple) and c[0] == "timestamp"
            ]
            if ts_candidates:
                ts_col = ts_candidates[0]
                ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                if ts.isna().any():
                    raise ValueError("Bad timestamps in portfolio CSV.")
                df = df.drop(columns=[ts_col])
                df.index = pd.DatetimeIndex(ts, name="timestamp")
                # Numeric coercion
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(how="any")
                # Ensure float dtypes
                df = df.astype("float64")
                return df.sort_index()
    except Exception:
        pass

    # Flat header fallback
    df2 = pd.read_csv(p)
    if "timestamp" not in df2.columns:
        raise ValueError("Portfolio CSV missing 'timestamp' column.")
    ts = pd.to_datetime(df2.pop("timestamp"), utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Bad timestamps in portfolio CSV.")
    df2.index = pd.DatetimeIndex(ts, name="timestamp")
    # Try to split columns like 'ASSET.close' into a MultiIndex
    if any("." in c for c in df2.columns):
        tuples = [tuple(c.split(".", 1)) for c in df2.columns]
        mi = pd.MultiIndex.from_tuples(tuples, names=["symbol", "field"])
        df2.columns = mi
    # numeric coercion
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.dropna(how="any").astype("float64")
    return df2.sort_index()


# --- replace in run.py ---


def _load_prices_from_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Robustly load either:
      - Single-asset OHLCV CSV → flat columns (open/high/low/close/volume)
      - Portfolio CSV → MultiIndex columns (symbol, field) or flat 'ASSET.close'
    Preference order:
      1) Single-asset OHLCV (strict schema)
      2) Portfolio (official loader, if available)
      3) Portfolio (generic fallback)
    """
    # 1) Try strict single-asset OHLCV first — this avoids misdetecting
    #    simple CSVs as degenerate MultiIndex.
    try:
        scheme = DataScheme(path=csv_path)
        df_ohlc = load_ohlcv_csv(scheme)
        return df_ohlc
    except Exception:
        pass

    # 2) Portfolio loader (if exported)
    if load_portfolio_csv is not None:
        try:
            return load_portfolio_csv(csv_path)  # type: ignore[misc]
        except Exception:
            pass

    # 3) Generic portfolio fallback
    return _generic_load_portfolio_csv(csv_path)


def _generic_load_portfolio_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a 'portfolio_<code>.csv' written with MultiIndex columns
    (two header rows) OR a flat form like 'BTCUSDT.close'.

    Returns a DataFrame with tz-aware UTC DatetimeIndex named 'timestamp'.
    """
    p = Path(path)

    # Helper: detect a *real* two-level header (not degenerate)
    def _valid_multiindex_columns(cols: pd.Index) -> bool:
        if not isinstance(cols, pd.MultiIndex):
            return False
        # level-1 must be meaningful (not all NaN/empty/‘Unnamed’)
        lvl1 = pd.Index(
            [str(x) if x is not None else "" for x in cols.get_level_values(1)]
        )
        if all((x == "") or x.lower().startswith("unnamed") for x in lvl1):
            return False
        # level-0 must contain 'timestamp' and at least one *symbol* besides it
        lvl0 = pd.Index(
            [str(x) if x is not None else "" for x in cols.get_level_values(0)]
        )
        has_ts = any(x.lower() == "timestamp" for x in lvl0)
        has_sym = any((x != "") and (x.lower() != "timestamp") for x in lvl0)
        return has_ts and has_sym

    # Try two-row header first, but only accept if it's a *valid* portfolio grid.
    try:
        df2 = pd.read_csv(p, header=[0, 1])
        if _valid_multiindex_columns(df2.columns):
            # find the timestamp column at level 0
            ts_candidates = [
                c
                for c in df2.columns
                if isinstance(c, tuple) and str(c[0]).lower() == "timestamp"
            ]
            if ts_candidates:
                ts_col = ts_candidates[0]
                ts = pd.to_datetime(df2[ts_col], utc=True, errors="coerce")
                if ts.isna().any():
                    raise ValueError("Bad timestamps in portfolio CSV.")
                df2 = df2.drop(columns=[ts_col])
                df2.index = pd.DatetimeIndex(ts, name="timestamp")
                # numeric coercion
                for c in df2.columns:
                    df2[c] = pd.to_numeric(df2[c], errors="coerce")
                df2 = df2.dropna(how="any").astype("float64")
                return df2.sort_index()
        # else: fall through to flat header parsing
    except Exception:
        pass

    # Flat header fallback (expects 'timestamp' + per-asset columns or 'ASSET.field')
    df = pd.read_csv(p)
    if "timestamp" not in df.columns:
        raise ValueError("Portfolio CSV missing 'timestamp' column.")
    ts = pd.to_datetime(df.pop("timestamp"), utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Bad timestamps in portfolio CSV.")
    df.index = pd.DatetimeIndex(ts, name="timestamp")

    # If columns look like 'ASSET.field', promote to MultiIndex
    if any("." in str(c) for c in df.columns):
        tuples = [tuple(str(c).split(".", 1)) for c in df.columns]
        mi = pd.MultiIndex.from_tuples(tuples, names=["symbol", "field"])
        df.columns = mi

    # numeric coercion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="any").astype("float64")
    return df.sort_index()


def _is_single_asset_ohlcv(df: pd.DataFrame) -> bool:
    cols = (
        set(map(str.lower, df.columns))
        if not isinstance(df.columns, pd.MultiIndex)
        else set()
    )
    return {"open", "high", "low", "close", "volume"}.issubset(cols)


def _extract_single_asset_ohlcv(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    From a portfolio DataFrame (MultiIndex), extract OHLCV for `asset`.
    If only (asset,'close') exists, synthesize minimal OHLC around close.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        # Already single-asset OHLCV? Return as-is (validated by caller).
        return df

    sym = str(asset).upper()
    have_fields = [
        f for f in ("open", "high", "low", "close", "volume") if (sym, f) in df.columns
    ]
    if have_fields:
        sub = df.loc[:, pd.MultiIndex.from_product([[sym], have_fields])]
        sub.columns = [f for _, f in sub.columns]  # flatten fields
        # pad missing
        for c in ("open", "high", "low", "close", "volume"):
            if c not in sub.columns:
                if c == "volume":
                    sub[c] = 0.0
                else:
                    close = sub["close"].astype(float)
                    if c == "open":
                        sub[c] = close.shift(1).fillna(close.iloc[0])
                    elif c == "high":
                        sub[c] = np.maximum(sub.get("open", close), close)
                    elif c == "low":
                        sub[c] = np.minimum(sub.get("open", close), close)
        sub = sub[["open", "high", "low", "close", "volume"]].astype("float64")
        sub.index = pd.DatetimeIndex(df.index, name="timestamp")
        return sub

    # only close available?
    if (sym, "close") not in df.columns:
        raise KeyError(
            f"{sym}: not found in portfolio frame (need {sym}.close or OHLCV)."
        )
    c = df[(sym, "close")].astype(float)
    o = c.shift(1).fillna(c.iloc[0])
    h = np.maximum(o, c)
    l = np.minimum(o, c)
    v = pd.Series(0.0, index=c.index)
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    out.index = pd.DatetimeIndex(df.index, name="timestamp")
    return out.astype("float64")


def _infer_universe(strategy: BaseStrategy, df: pd.DataFrame) -> List[str]:
    """
    Determine training universe for a multi-asset strategy.
    Priority:
      1) strategy.assets or strategy.universe or strategy.symbols  (first found)
      2) All symbols in df (if MultiIndex)
    """
    for attr in ("assets", "universe", "symbols"):
        if hasattr(strategy, attr):
            val = getattr(strategy, attr)
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return [str(x).upper() for x in val]
    if isinstance(df.columns, pd.MultiIndex) and "symbol" in (df.columns.names or []):
        return sorted(
            list({str(s).upper() for s in df.columns.get_level_values("symbol")})
        )
    raise ValueError(
        "Cannot infer multi-asset universe: provide a portfolio CSV with MultiIndex columns "
        "or set `strategy.assets` / `strategy.universe` explicitly."
    )


def _close_panel_from_df(df: pd.DataFrame, universe: Sequence[str]) -> pd.DataFrame:
    """
    Build a (T, M) close-price panel for a list of assets.
    Accepts portfolio MultiIndex df or a flat per-asset price table.
    """
    if isinstance(df.columns, pd.MultiIndex):
        missing = [(a, "close") for a in universe if (a, "close") not in df.columns]
        if missing:
            raise KeyError(f"Missing close columns for: {missing}")
        P = pd.concat([df[(a, "close")].rename(a) for a in universe], axis=1)
        return P.astype("float64")

    # Flat: if it's single-asset OHLCV, require len(universe)==1
    if _is_single_asset_ohlcv(df):
        if len(universe) != 1:
            raise ValueError(
                "Single-asset OHLCV provided but multi-asset universe requested. "
                "Either pass a portfolio CSV or restrict the universe to one asset."
            )
        return pd.DataFrame(
            {universe[0]: df["close"].astype("float64")}, index=df.index
        )

    # Otherwise expect per-asset columns already (e.g., ['BTCUSDT','ETHUSDT',...])
    for a in universe:
        if a not in df.columns:
            raise KeyError(f"Missing price column for asset: {a}")
    return df.loc[:, list(universe)].astype("float64")


def _estimate_bars_per_year(idx: pd.DatetimeIndex) -> float:
    """Estimate bars-per-year from median spacing."""
    if len(idx) < 2:
        return 252.0
    deltas = idx.to_series().diff().dropna().dt.total_seconds()
    if len(deltas) == 0:
        return 252.0
    med = deltas.median()
    if med <= 0:
        return 252.0
    sec_per_year = 365.0 * 24.0 * 3600.0
    return float(sec_per_year / med)


# ---------------------------------------------------------------------
# Public results
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TrainResult:
    strategy: BaseStrategy
    fit: FitResultLearning
    model_path: Path
    data_meta: Dict[str, Any]


@dataclass(frozen=True)
class RunResult:
    weights: pd.DataFrame
    holdings: pd.DataFrame
    equity: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]


# ---------------------------------------------------------------------
# Training workflow (single-asset OR multi-asset)
# ---------------------------------------------------------------------


def train_on_csv(
    csv_path: Union[str, Path],
    strategy_name: Union[str, BaseStrategy],
    *,
    strategy_params: Optional[Mapping[str, Any]] = None,
    learning: Optional[Union[LearningConfig, Mapping[str, Any]]] = None,
    optimization: Optional[Union[OptimizationConfig, Mapping[str, Any]]] = None,
    model_dir: Union[str, Path] = "models",
    code_hint: Optional[str] = None,
) -> TrainResult:
    """
    Train (fit) a strategy on a CSV and save the model.

    Single-asset strategies:
      - Accept either single-asset OHLCV CSV or portfolio CSV.
      - For portfolio CSVs, extract/synthesize OHLCV for `strategy.asset`.
      - Build asset_returns from close.

    Multi-asset strategies:
      - Accept portfolio CSV (MultiIndex preferred). Single-asset OHLCV allowed
        only if the inferred universe has exactly one asset.
      - Universe taken from strategy (assets/universe/symbols) or from data.
      - Build a close-price panel and asset_returns as pct_change.

    Implementation detail:
      - To avoid RF broadcast bugs in portfolio alignment, we *append a CASH
        return column* (last) to `asset_returns` and set
        `learning.has_cash_in_returns = True`, so the loss path will not try
        to inject RF again.
    """

    # -------------------------- local helpers --------------------------
    def _is_single_asset_ohlcv(df: pd.DataFrame) -> bool:
        cols = set(map(str, df.columns))
        return (not isinstance(df.columns, pd.MultiIndex)) and {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }.issubset(cols)

    def _extract_single_asset_ohlcv(df: pd.DataFrame, asset: str) -> pd.DataFrame:
        a = str(asset).upper()
        if isinstance(df.columns, pd.MultiIndex):
            if (a, "close") not in df.columns:
                raise KeyError(f"{a}: not found (need {a}.close).")
            c = df[(a, "close")].astype(float)
        else:
            if a in df.columns and "close" not in df.columns:
                c = pd.to_numeric(df[a], errors="coerce").astype(float)
            else:
                raise KeyError(f"{a}: not found (need {a}.close or an OHLCV CSV).")
        o = c.shift(1).fillna(c.iloc[0])
        h = c
        l = c
        v = pd.Series(0.0, index=c.index)
        out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
        out.index = df.index
        return out.astype("float64")

    def _infer_universe(strategy: BaseStrategy, df: pd.DataFrame) -> list[str]:
        for attr in ("assets", "universe", "symbols"):
            v = getattr(strategy, attr, None)
            if v is not None:
                seq = list(v) if isinstance(v, (list, tuple, set)) else [v]
                uni = [str(s).upper() for s in seq]
                if uni:
                    return uni
        if isinstance(df.columns, pd.MultiIndex):
            return [str(s).upper() for s in df.columns.get_level_values(0).unique()]
        a = getattr(strategy, "asset", None)
        if a:
            return [str(a).upper()]
        raise ValueError(
            "Cannot infer universe — provide strategy.assets/universe/symbols or use a portfolio CSV."
        )

    def _close_panel_from_df(df: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            missing = [(s, "close") for s in universe if (s, "close") not in df.columns]
            if missing:
                raise KeyError(f"Missing close columns for: {missing}")
            return pd.concat(
                [df[(s, "close")].rename(s).astype(float) for s in universe], axis=1
            )
        if _is_single_asset_ohlcv(df):
            if len(universe) != 1:
                raise ValueError(
                    "Single-asset OHLCV provided but multi-asset universe requested."
                )
            return pd.DataFrame(
                {universe[0]: df["close"].astype(float)}, index=df.index
            )
        miss = [s for s in universe if s not in df.columns]
        if miss:
            raise KeyError(f"Missing price columns for: {miss}")
        return df[universe].astype(float)

    def _rf_scalar(x: Any) -> float:
        import numpy as _np, pandas as _pd

        if x is None:
            return 0.0
        if _np.isscalar(x):
            return float(x)
        if isinstance(x, (_pd.Series, _pd.DataFrame, list, tuple, _np.ndarray)):
            arr = _np.asarray(x)
            if arr.size == 0:
                return 0.0
            return float(arr.mean())  # deterministic collapse
        try:
            return float(x)
        except Exception:
            return 0.0

    # ---------------------- load & normalize data ----------------------
    df_raw = _load_prices_from_csv(csv_path)
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        raise TypeError("CSV must yield a DatetimeIndex.")
    if df_raw.index.tz is None:
        df_raw.index = df_raw.index.tz_localize("UTC")

    # Resolve strategy
    strategy_params = dict(strategy_params or {})
    strategy = _resolve_strategy(strategy_name, **strategy_params)

    # --------------- frame for fit + risky returns panel ---------------
    if getattr(strategy, "is_multi_asset", None) and strategy.is_multi_asset():
        universe = _infer_universe(strategy, df_raw)
        P = _close_panel_from_df(df_raw, universe=universe).astype("float64")
        asset_returns = P.pct_change().fillna(0.0).astype("float64")
        df_for_fit = df_raw
        data_universe = list(P.columns)
    else:
        if not hasattr(strategy, "asset"):
            raise AttributeError(
                "Single-asset training requires the strategy to expose `.asset`."
            )
        asset = str(getattr(strategy, "asset")).upper()
        if _is_single_asset_ohlcv(df_raw):
            df_for_fit = df_raw.astype("float64")
        else:
            df_for_fit = _extract_single_asset_ohlcv(df_raw, asset=asset)
        asset_returns = pd.DataFrame(
            {asset: df_for_fit["close"].pct_change().fillna(0.0).astype(float)},
            index=df_for_fit.index,
        ).astype("float64")
        data_universe = [asset]

    # ---------------- learning / optimization configs -----------------
    if learning is None:
        learning = LearningConfig()
    elif isinstance(learning, Mapping):
        learning = LearningConfig(**learning)  # type: ignore[arg-type]

    if optimization is None:
        optimization = OptimizationConfig()
    elif isinstance(optimization, Mapping):
        optimization = OptimizationConfig(**optimization)  # type: ignore[arg-type]

    # ---------------------- RF handling (scalar) -----------------------
    # Determine one scalar RF to use (applies also to CASH column below).
    rf_candidates = []
    if hasattr(learning, "rf"):
        rf_candidates.append(getattr(learning, "rf"))
    if hasattr(learning, "rf_per_period"):
        rf_candidates.append(getattr(learning, "rf_per_period"))
    if hasattr(learning, "objective") and hasattr(learning.objective, "rf_per_period"):
        rf_candidates.append(getattr(learning.objective, "rf_per_period"))
    rf_scalar = _rf_scalar(next((x for x in rf_candidates if x is not None), 0.0))

    # Normalize on the objects (keep them scalar to avoid later broadcast paths)
    try:
        if hasattr(learning, "rf"):
            setattr(learning, "rf", rf_scalar)
        if hasattr(learning, "rf_per_period"):
            setattr(learning, "rf_per_period", rf_scalar)
        if hasattr(learning, "objective") and hasattr(
            learning.objective, "rf_per_period"
        ):
            setattr(learning.objective, "rf_per_period", rf_scalar)
    except Exception:
        pass

    # ---------------- include CASH in asset_returns --------------------
    # Make CASH the *last* column, set has_cash_in_returns=True to skip RF injection.
    cash_series = pd.Series(
        rf_scalar, index=asset_returns.index, name=CASH_COL, dtype="float64"
    )
    asset_returns = pd.concat([asset_returns, cash_series], axis=1).astype("float64")

    # If LearningConfig exposes these fields, set them (safe no-ops otherwise)
    if hasattr(learning, "has_cash_in_returns"):
        setattr(learning, "has_cash_in_returns", True)
    if hasattr(learning, "cash_col_idx"):
        setattr(
            learning, "cash_col_idx", None
        )  # None → treat last column as CASH downstream

    # ------------------------------ fit -------------------------------
    fit = fit_strategy(
        strategy=strategy,
        df=df_for_fit,
        asset_returns=asset_returns,
        learning=learning,
        optimization=optimization,
    )

    # --------------------------- save model ---------------------------
    data_meta = {
        "csv_path": str(Path(csv_path).resolve()),
        "n_bars": int(len(df_for_fit)),
        "index_start": df_for_fit.index[0].isoformat() if len(df_for_fit) else None,
        "index_end": df_for_fit.index[-1].isoformat() if len(df_for_fit) else None,
        "columns": (
            list(df_for_fit.columns)
            if not isinstance(df_for_fit.columns, pd.MultiIndex)
            else [tuple(map(str, t)) for t in df_for_fit.columns]
        ),
        "universe": list(map(str, data_universe)),
        "is_multi_asset": bool(getattr(strategy, "is_multi_asset", lambda: False)()),
        "code": code_hint or Path(csv_path).stem,
    }
    payload = _model_payload(strategy, fit, data_meta=data_meta)
    model_path = _save_model_json(payload, out_dir=model_dir, code_hint=code_hint)

    return TrainResult(
        strategy=strategy, fit=fit, model_path=model_path, data_meta=data_meta
    )


# ---------------------------------------------------------------------
# Inference / evaluation workflow (unchanged)
# ---------------------------------------------------------------------


def _coerce_strategy(strategy_or_path: Union[str, Path, BaseStrategy]) -> BaseStrategy:
    """Accept a strategy instance or a path to a saved model."""
    if isinstance(strategy_or_path, BaseStrategy):
        return strategy_or_path
    p = Path(strategy_or_path)
    if p.exists() and p.suffix.lower() == ".json":
        return load_model(p)
    return _resolve_strategy(str(strategy_or_path))


def _build_prices_panel_for_weights(
    df: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    cash_col: str = CASH_COL,
    field: str = "close",
) -> pd.DataFrame:
    """
    Construct a prices frame with columns for each risky asset in `weights`.
    Supports single-asset OHLCV and MultiIndex portfolio DataFrames.
    """
    risky = [c for c in weights.columns if c != cash_col]

    if isinstance(df.columns, pd.MultiIndex):
        # Expect (asset, field) style columns
        missing = [(a, field) for a in risky if (a, field) not in df.columns]
        if missing:
            raise KeyError(f"Missing price columns in input: {missing}")
        P = pd.concat([df[(a, field)].rename(a) for a in risky], axis=1)
    else:
        # Single-asset or flat portfolio with separate columns per asset
        if set(["open", "high", "low", "close", "volume"]).issubset(df.columns):
            if len(risky) != 1:
                raise ValueError(
                    "Single-asset OHLCV detected, but weights have multiple risky assets."
                )
            P = pd.DataFrame({risky[0]: df["close"]}, index=df.index)
        else:
            missing = [a for a in risky if a not in df.columns]
            if missing:
                raise KeyError(f"Missing price columns for risky assets: {missing}")
            P = df[risky].copy()

    if (P <= 0).any().any():
        raise ValueError("Non-positive prices encountered.")
    return P


def run_on_csv(
    csv_path: Union[str, Path],
    strategy_or_model: Union[str, Path, BaseStrategy],
    *,
    metrics: Sequence[str] = ("roi", "cagr", "sharpe", "max_drawdown", "calmar"),
    rf: Union[float, pd.Series] = 0.0,
    init_equity: float = 1.0,
) -> RunResult:
    """
    Evaluate a strategy on a CSV: weights → holdings → equity → metrics.

    Implementation detail:
    - We explicitly construct an asset-returns matrix that ALREADY includes the
      CASH leg as a 1-D vector (shape (T,)), aligned to the weights column order.
      Then we call `portfolio_returns(..., has_cash_in_returns=True)` which
      bypasses the RF-injection branch that causes the (T,1) → (T,) broadcast error.
    """
    df = _load_prices_from_csv(csv_path)
    # Ensure tz-aware UTC index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("CSV must yield a DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    strat = _coerce_strategy(strategy_or_model)

    # Produce weights
    res = strat.predict_weights(df)
    if isinstance(res, StrategyResult):
        W = res.weights
        artifacts = dict(res.artifacts or {})
    else:
        W = res  # type: ignore[assignment]
        artifacts = {}
    if CASH_COL not in W.columns:
        raise KeyError(f"Weights must include a '{CASH_COL}' column.")

    # Build prices panel in risky order expected by weights_to_holdings_df
    P = _build_prices_panel_for_weights(df, W, cash_col=CASH_COL, field="close")

    # Holdings (units) + equity
    H: HoldingsResult = weights_to_holdings_df(
        weights=W,
        prices=P,
        cash_col=CASH_COL,
        field="close",
        init_equity=float(init_equity),
        asset_returns=None,  # derive from prices + rf
        rf=rf,
        has_cash_in_returns=None,
    )
    holdings, equity_ser, trades = H.holdings, H.equity, H.trades

    # ---------- Metrics: build returns INCLUDING CASH as the last/placed column ----------
    risky_returns = P.pct_change().fillna(0.0)

    T = len(W)
    # rf as a 1-D vector aligned to W.index
    if isinstance(rf, pd.Series):
        rf_vec = rf.reindex(W.index).astype(float).fillna(0.0).to_numpy(dtype=float)
    else:
        rf_vec = np.full(T, float(rf), dtype=float)

    # Assemble returns matrix in the SAME column order as W (incl. CASH)
    ret_cols: list[np.ndarray] = []
    for col in W.columns:
        if col == CASH_COL:
            ret_cols.append(rf_vec)  # 1-D (T,)
        else:
            s = (
                risky_returns[col]
                .reindex(W.index)
                .astype(float)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            ret_cols.append(s)

    R_full = np.column_stack(ret_cols)  # shape (T, M), includes CASH

    rp = portfolio_returns(
        weights=W.to_numpy(dtype=float),
        asset_returns=R_full,
        rf=0.0,  # ignored because has_cash_in_returns=True
        has_cash_in_returns=True,
        cash_col_idx=W.columns.get_loc(CASH_COL),
    )
    rp_ser = pd.Series(rp, index=W.index, name="port_ret")

    # Metrics
    bars_per_year = _estimate_bars_per_year(W.index)  # for Sharpe scaling
    metrics_out: Dict[str, Any] = {}
    for key in metrics:
        k = key.lower()
        if k == "roi":
            metrics_out["roi"] = roi(equity_ser)
        elif k == "cagr":
            metrics_out["cagr_years"] = cagr(equity_ser, time_unit="years")
        elif k == "sharpe":
            metrics_out["sharpe"] = sharpe(
                rp_ser, bars_per_unit=bars_per_year, rf=0.0, ddof=0
            )
        elif k in ("mdd", "max_drawdown"):
            mdd, peak_ts, trough_ts = max_drawdown(equity_ser)
            metrics_out["max_drawdown"] = mdd
            metrics_out["mdd_peak_ts"] = peak_ts
            metrics_out["mdd_trough_ts"] = trough_ts
        elif k in ("calmar", "calmar_nominal"):
            metrics_out["calmar_nominal"] = calmar(
                equity_ser, mode="nominal", rf=0.0, time_unit="years"
            )
        elif k == "hit_rate":
            metrics_out["hit_rate"] = hit_rate(rp_ser)
        else:
            pass

    artifacts.update(
        {
            "prices": P,
            "port_returns": rp_ser,
            "bars_per_year": bars_per_year,
        }
    )

    return RunResult(
        weights=W,
        holdings=holdings,
        equity=equity_ser,
        trades=trades,
        metrics=metrics_out,
        artifacts=artifacts,
    )

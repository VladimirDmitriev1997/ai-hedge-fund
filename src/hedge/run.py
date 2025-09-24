# run.py
"""
High-level runners for training and evaluating strategies on CSV data.

This module exposes two workflows:

1) train_on_csv(...)
   - Load an OHLCV CSV.
   - Initialize a strategy by name (from a registry) with given params.
   - Fit it with a chosen optimization mode and objective (via learning.py).
   - Save the fitted model + training metadata into ./models.
   - Return (strategy, FitResult, model_path).

2) run_on_csv(...)
   - Load an OHLCV (or portfolio) CSV.
   - Load a strategy (object or saved JSON path) or accept an instance.
   - Generate weights, convert to holdings, compute requested metrics.
   - Return a RunResult with weights, holdings, equity, trades, metrics.

Notes
-----
- Assumes single-asset CSVs by default (OHLCV columns). Multi-asset
  "portfolio" CSVs created by your data helpers are supported if you pass
  a prices DataFrame with columns per asset (or MultiIndex (asset, "close")).
- Risk-free 'rf' is optional; if provided as a per-bar Series aligned to the
  index, it is used in holdings conversion and Sharpe (see metrics).
"""


from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union, List
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Project imports
from hedge.data import DataScheme, load_ohlcv_csv, load_portfolio_csv


from hedge.strategies import (
    BaseStrategy,
    SingleAssetMACrossover,
    StrategyResult,
)
from hedge.learning import (
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
# Strategy
# ---------------------------------------------------------------------

# Map public names → constructors. Extend as you add new strategies.
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
# Model
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
    for k, cls in STRATEGY_REGISTRY.items():
        if cls.__name__ == cls_name:
            return cls.from_dict(payload["strategy_config"])
    # Fallback: dynamic import
    mod = __import__(module_name, fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    return cls.from_dict(payload["strategy_config"])


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _load_prices_from_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    # Try portfolio (multi-asset) loader first; fall back to single-asset OHLCV
    try:
        dfp = load_portfolio_csv(csv_path)
        return dfp
    except Exception:
        scheme = DataScheme(path=csv_path)
        return load_ohlcv_csv(scheme)


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


def _estimate_bars_per_year(idx: pd.DatetimeIndex) -> float:
    """
    Estimate bars-per-year from median spacing.

    """
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
# Training workflow
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
    Train (fit) a strategy on an OHLCV CSV and save the model.

    Parameters
    ----------
    csv_path : str | Path
        Path to CSV with columns: timestamp, open, high, low, close, volume.
    strategy_name : str | BaseStrategy
        Strategy key from registry or an already constructed instance.
    strategy_params : mapping, optional
        Params passed to the strategy constructor (if name provided).
    learning : LearningConfig | mapping, optional
        Objective/cost/RF configuration (per-bar rf if you provide one).
    optimization : OptimizationConfig | mapping, optional
        Optimization mode and hyperparameters.
    model_dir : str | Path, default "models"
        Directory to store saved models (JSON).
    code_hint : str, optional
        Extra suffix in saved filename (e.g., symbol or dataset code).

    Returns
    -------
    TrainResult
        (strategy, fit result, model path, data meta)
    """
    df = _load_prices_from_csv(csv_path)

    # Resolve strategy
    strategy_params = dict(strategy_params or {})
    strategy = _resolve_strategy(strategy_name, **strategy_params)

    # Build per-asset returns for training
    # Single-asset default: one risky leg called by strategy.asset
    if isinstance(df.columns, pd.MultiIndex):
        raise NotImplementedError(
            "Multi-asset training from a single CSV is not supported here. "
            "Use your portfolio data builder and pass returns accordingly."
        )

    # Risky returns: simple returns of close under column strategy.asset
    if not hasattr(strategy, "asset"):
        raise AttributeError(
            "Strategy must expose `.asset` for single-asset training with CSV."
        )
    risky_col_name = getattr(strategy, "asset")
    asset_returns = pd.DataFrame(
        {risky_col_name: df["close"].pct_change()},
        index=df.index,
    ).fillna(
        0.0
    )  # first bar 0

    # Learning / Optimization configs
    if learning is None:
        learning = LearningConfig()
    elif isinstance(learning, Mapping):
        learning = LearningConfig(**learning)  # type: ignore[arg-type]

    if optimization is None:
        optimization = OptimizationConfig()
    elif isinstance(optimization, Mapping):
        optimization = OptimizationConfig(**optimization)  # type: ignore[arg-type]

    # Fit
    fit = fit_strategy(
        strategy=strategy,
        df=df,
        asset_returns=asset_returns,
        learning=learning,
        optimization=optimization,
    )

    # Save model
    data_meta = {
        "csv_path": str(Path(csv_path).resolve()),
        "n_bars": int(len(df)),
        "index_start": df.index[0].isoformat() if len(df) else None,
        "index_end": df.index[-1].isoformat() if len(df) else None,
        "columns": list(df.columns),
        "code": code_hint or Path(csv_path).stem,
    }
    payload = _model_payload(strategy, fit, data_meta=data_meta)
    model_path = _save_model_json(payload, out_dir=model_dir, code_hint=code_hint)

    return TrainResult(
        strategy=strategy, fit=fit, model_path=model_path, data_meta=data_meta
    )


# ---------------------------------------------------------------------
# Inference / evaluation workflow
# ---------------------------------------------------------------------


def _coerce_strategy(strategy_or_path: Union[str, Path, BaseStrategy]) -> BaseStrategy:
    """
    Accept a strategy instance or a path to a saved model.
    """
    if isinstance(strategy_or_path, BaseStrategy):
        return strategy_or_path
    p = Path(strategy_or_path)
    if p.exists() and p.suffix.lower() == ".json":
        return load_model(p)
    # else resolve by registry key
    return _resolve_strategy(str(strategy_or_path))


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

    Parameters
    ----------
    csv_path : str | Path
        Path to OHLCV (or portfolio) CSV.
    strategy_or_model : str | Path | BaseStrategy
        Strategy instance, registry key, or path to saved JSON.
    metrics : sequence of str, default ('roi','cagr','sharpe','max_drawdown','calmar')
        Metric keys to compute.
    rf : float | pd.Series, default 0.0
        Per-period risk-free (per bar), aligned to the CSV index if Series.
    init_equity : float, default 1.0
        Initial equity for holdings sizing.

    Returns
    -------
    RunResult
        weights, holdings, equity, trades, metrics, artifacts
    """
    df = _load_prices_from_csv(csv_path)
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
    H = weights_to_holdings_df(
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

    # Portfolio returns from weights + asset returns (derived from prices)
    risky_returns = P.pct_change().fillna(0.0)
    rp = portfolio_returns(
        weights=W.to_numpy(dtype=float),
        asset_returns=risky_returns.to_numpy(dtype=float),
        rf=(rf.to_numpy() if isinstance(rf, pd.Series) else rf),
        has_cash_in_returns=False,
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
            # metrics.sharpe expects rf per chosen unit; we pass 0.0 by default.
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
            # ignored unknown keys to keep robust; could raise instead
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

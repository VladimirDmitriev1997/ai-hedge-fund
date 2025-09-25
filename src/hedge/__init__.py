"""
hedge – compact toolkit for research-grade trading backtests.

Public surface:
- Data: DataScheme, load_ohlcv_csv, resample_ohlcv, load_portfolio_csv
- Features: sma, ema, rsi, macd, atr, logret, zscore (as available)
- Portfolio math: CASH_COL, portfolio_returns, weights_to_holdings_df, HoldingsResult
- Strategies: StrategyResult, SingleAssetMACrossover
- Fitting: LearningConfig, OptimizationConfig, FitResult, fit_strategy, Objective
- Metrics: roi, cagr, sharpe, max_drawdown, calmar, hit_rate, equity_curve
"""

from importlib import metadata

# ----- version -----
try:
    __version__ = metadata.version("ai-hedge-fund")
except Exception:  # fallback in editable dev
    __version__ = "0.0.0"

# ----- re-exports -----
from .data import (
    DataScheme,
    load_ohlcv_csv,
    resample_ohlcv,
    load_portfolio_csv,
    build_and_save_portfolio_dataset,
)
from .features import *  # keep simple; your module defines the canonical names
from .portfolio import (
    CASH_COL,
    portfolio_returns,
    weights_to_holdings_df,
    HoldingsResult,
)
from .strategies import StrategyResult, SingleAssetMACrossover
from .metrics import (
    roi,
    cagr,
    sharpe,
    max_drawdown,
    calmar,
    hit_rate,
    equity_curve,
)
from .fitting.learning import (
    LearningConfig,
    OptimizationConfig,
    FitResult,
    fit_strategy,
)
from .fitting.losses import Objective, evaluate_objective

__all__ = [
    "DataScheme",
    "load_ohlcv_csv",
    "resample_ohlcv",
    "load_portfolio_csv",
    "build_and_save_portfolio_dataset",
    "CASH_COL",
    "portfolio_returns",
    "weights_to_holdings_df",
    "HoldingsResult",
    "StrategyResult",
    "SingleAssetMACrossover",
    "LearningConfig",
    "OptimizationConfig",
    "FitResult",
    "fit_strategy",
    "Objective",
    "evaluate_objective",
    "roi",
    "cagr",
    "sharpe",
    "max_drawdown",
    "calmar",
    "hit_rate",
    "equity_curve",
]

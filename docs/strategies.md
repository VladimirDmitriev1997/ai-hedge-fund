# Strategy Library — Weights, Self-Financing & Baselines

This module defines **weights-producing** strategies for OHLCV time series.  
Each strategy outputs a time-indexed DataFrame of **portfolio weights** that sum to `1.0` per row, including an explicit `"CASH"` column. We keep everything **causal** (decisions at `t` apply from `t + trade_lag_bars`) and deterministic.

## Contents
- [Notation](#notation)
- [Self-Financing in Weights](#self-financing-in-weights)
  - [Wealth recursion](#wealth-recursion)
  - [Execution (code-level view)](#execution-code-level-view)
- [Design & Interfaces](#design--interfaces)
- [Helpers](#helpers)
- [Baseline Strategy](#baseline-strategy-single-asset-sma-crossover)
- [Future Strategies (Skeletons)](#future-strategies-skeletons)
- [References](#references)
- [Usage](#usage)

---

## Notation

- `W_t` — portfolio wealth right **before** rebalancing at time `t`.
- `h_t` — vector of risky **holdings** before rebalance at `t`.
- `P_t` — vector of execution prices used at `t` (e.g., next-bar open).
- `C_t` — **cash** balance before rebalance at `t`.
- `w_t` — **post-trade risky weights** applied over `[t, t+1]`.
- `w_t_cash` — **post-trade cash weight** over `[t, t+1]`.
- `r_{t+1}` — risky asset **gross** returns over `[t, t+1]` (fractional).
- `r_f_{t,t+1}` — cash (risk-free) return over `[t, t+1]` (fractional).
- `K_t` — fees/impact paid at `t`.

Weights are defined post-trade:

- `w_t := (h_t^+ ⊙ P_t) / W_t`
- `w_t_cash := C_t^+ / W_t`

Budget (post-trade): `sum_i w_{t,i} + w_t_cash = 1`.

---

## Self-Financing in Weights

If a policy (sizer) **updates cash** to pay for rebalances at `P_t` (including fees) and enforces the **budget constraint** above, then the portfolio is **self-financing** (no external cash injections).

> **Important:** Having `sum_i w_{t,i} = 1` **without** cash (or without updating it at execution prices) is **not** sufficient. The self-financing guarantee is the **trade budget equation**: sales fund purchases and fees, and the remainder is accounted for in `C_t^+`.

### Wealth recursion

With weights including cash:

`W_{t+1} / W_t = 1 + w_t^T r_{t+1} + w_t_cash * r_f_{t,t+1} - K_t / W_t`


If you fold cash into the asset list (treat cash as asset `0` with return `r_f`), you get:

`W_{t+1} / W_t = 1 + w̃_t^T r̃_{t+1} - K_t / W_t`,
`with sum_j w̃_{t,j} = 1`.


### Execution (code-level view)

At each rebalance `t`:

1. **Decide weights** `w_t` (including `"CASH"`).
2. **Target dollars**: `H_target_t = w_t * W_t`.
3. **Trades at prices `P_t`**: `Δh_t = H_target_t / P_t - h_t`.
4. **Trade value**: `trade_value = (Δh_t)^T P_t` (buys positive).
5. **Fees/impact**: e.g., `K_t = κ^T |Δh_t ⊙ P_t|`.
6. **Update cash**: `C_t^+ = C_t - trade_value - K_t`.
7. **Update holdings**: `h_t^+ = h_t + Δh_t`.

This sequence makes the strategy **exactly** self-financing.

---

## Design & Interfaces

Strategies are **dataclasses** with JSON-friendly parameters/state.  
Each strategy exposes:

- `fit(df) -> self` — optional; for calibration/learning later.
- `predict_weights(df) -> StrategyResult` — returns a **causal** weights DataFrame whose rows sum to `1.0` (including `"CASH"`), plus `artifacts` (diagnostics) and small `meta`.

### Causality

We adopt a **trade lag** `trade_lag_bars`:

- Decide at the close of `t`, **apply** from `t + trade_lag_bars` (default `1`).
- Implemented by shifting the weight matrix forward and filling the leading rows with neutral allocation (all `"CASH"`).

---

## Helpers

We use a few small helpers (already implemented in `strategies.py`):

- `_normalize_simplex_with_cash(W_risky, cash_col="CASH")`  
  From risky weights (may have NaNs), build `[risky..., CASH]` where each row sums to `1.0`.  
  If a row has no signal (all NaN), allocate `100%` to `"CASH"`.

- `_apply_causal_lag(W, lag)`  
  Shift the weight matrix forward by `lag` bars; fill the new leading rows with neutral allocation (all `"CASH"`).

- `_enforce_row_sums_one(W, cash_col="CASH")`  
  Minimal post-lag fixups: handle edge cases (NaN rows), recompute `"CASH"` as residual `1 - sum(risky)`, validate totals.

---

## Baseline Strategy: Single-Asset SMA Crossover

A simple, robust baseline that maps **SMA(fast) vs SMA(slow)** into weights on `[ASSET, CASH]`.

### Theory (discrete regime)

- Compute `SMA_fast` and `SMA_slow` on the chosen price (e.g., `close`).
- **Signal** `s_t` in `{-1, 0, +1}`:
  - `+1` if `SMA_fast > SMA_slow`
  - `-1` if `SMA_fast < SMA_slow`
  - `0` otherwise (or during warm-up)
- **Risky weight** `w_asset_t`:
  - `w_long` when `s_t = +1`
  - `w_short` when `s_t = -1` (default `0` for “no shorting”)
  - `0` when neutral/warm-up
- **Cash weight**: `w_cash_t = 1 - w_asset_t`.

This yields a **fully invested** (or long-flat, if `w_short=0`) portfolio, with `row_sum = 1`.

### Implementation details

- **Causality:** weights are shifted by `trade_lag_bars` (default `1`).
- **Warm-up:** first `slow_window` bars are set to neutral (`100%` in `"CASH"`).
- **Clipping:** per-asset clip is applied on the risky weight before adding `"CASH"`.

### Parameters

- `asset` — column name used in the weights matrix (e.g., `"BTCUSDT"`).
- `price_field` — which price to use (e.g., `"close"`).
- `fast_window`, `slow_window` — SMA lookbacks.
- `w_long`, `w_short` — risky allocation in the two regimes.
- `trade_lag_bars` — causal lag (usually `1`).
- `clip_per_asset` — bound for the risky weight before adding `"CASH"`.

**Output**

- `weights` with columns `[asset, "CASH"]`, rows sum to `1.0`.
- `artifacts`: `sma_fast`, `sma_slow`, `raw_signal`, `raw_weight_asset`.
- `meta`: parameters and bookkeeping.

---

## Future Strategies (Skeletons)

We include extensible skeletons that follow the **same weights contract**:

- **Cross-Sectional Momentum (multi-asset)**  
  Rank assets by past returns over a `lookback`; allocate equally to top-`k` (or proportionally), remainder to `"CASH"`. Output columns: `[asset_1, ..., asset_n, "CASH"]`.

- **Ensemble of Weights**  
  Combine multiple strategies’ weight matrices (aligned, weighted sum), then re-normalize to `[...,"CASH"]`, apply lag, and enforce row sums.

All strategies return causal weights with a `"CASH"` column and per-row sum `1.0`, making them directly usable by a simple **self-financing** backtester.

---

## References

**General & mathy**
- Grinold, R., Kahn, R. (2000). *Active Portfolio Management.*  
- Luenberger, D. (1997). *Investment Science.*  
- López de Prado, M. (2018). *Advances in Financial Machine Learning.*

**Baselines**
- Moving Average — <https://en.wikipedia.org/wiki/Moving_average>  
- Donchian Channel / Breakouts — <https://en.wikipedia.org/wiki/Donchian_channel>  
- Momentum — <https://en.wikipedia.org/wiki/Momentum_(finance)>

---

## Usage

```python
import pandas as pd

from hedge.data import load_ohlcv_csv, DataScheme
from hedge.strategies import SingleAssetMACrossover
from hedge.metrics import simple_returns, equity_curve, sharpe, max_drawdown

# 1) Load data (single asset)
df = load_ohlcv_csv(DataScheme(path="data/BTCUSDT_1h.csv"))

# 2) Configure baseline strategy (BTC long/flat crossover)
strat = SingleAssetMACrossover(
    asset="BTCUSDT",
    price_field="close",
    fast_window=20,
    slow_window=50,
    w_long=1.0,     # fully long in up regime
    w_short=0.0,    # flat in down regime
    trade_lag_bars=1
)

# 3) Produce causal weights on [BTCUSDT, CASH]
res = strat.predict_weights(df)
W = res.weights   # columns: ["BTCUSDT", "CASH"], each row sums to 1.0

# 4) Convert weights to returns (example: use simple close-to-close asset returns)
#    r_asset_t = C_t / C_{t-1} - 1
r_asset = df["close"].pct_change().fillna(0.0)

# 5) Portfolio returns with cash at 0% (set cash return if you have it)
r_cash = 0.0
r_port = W["BTCUSDT"] * r_asset + W["CASH"] * r_cash

# 6) Equity & metrics
eq = equity_curve(r_port, init_equity=1.0)
sr = sharpe(r_port, bars_per_year=24*365, rf_annual=0.0)
mdd, t_peak, t_trough = max_drawdown(eq)

print(f"Sharpe (1h bars): {sr:.2f}, MDD: {mdd:.1%}, Start: {eq.index[0].date()} → End: {eq.index[-1].date()}")

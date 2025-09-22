# Metrics Library — Definitions & Formulas

This module provides **vectorized, deterministic** performance metrics for backtests and strategies.

## Contents
- [Notation](#notation)
- [Equity & PnL](#equity--pnl)
  - [Equity Curve](#equity-curve)
  - [Per-bar PnL](#per-bar-pnl)
- [Return & Scaling Conventions](#return--scaling-conventions)
- [Metrics](#metrics)
  - [ROI (Total Return)](#roi-total-return)
  - [Excess ROI](#excess-roi)
  - [CAGR (per unit)](#cagr-per-unit)
  - [Excess CAGR (geometric)](#excess-cagr-geometric)
  - [Volatility (scaled)](#volatility-scaled)
  - [Sharpe Ratio](#sharpe-ratio)
  - [Maximum Drawdown (MDD)](#maximum-drawdown-mdd)
  - [Calmar (Nominal)](#calmar-nominal)
  - [Calmar (Excess, geometric)](#calmar-excess-geometric)
  - [Calmar (CAGR − Rf)](#calmar-cagr--rf)
  - [Hit Rate](#hit-rate)
  - [Turnover](#turnover)
- [References](#references)
- [Usage](#usage)

---

## Notation

- `r_t` — per-bar fractional return (e.g., `0.002` = `+0.2%`).
- `E_t` — equity (account value) at bar `t`.
- `idx` — `DatetimeIndex` of the series (`tz` optional).
- `elapsed_from_index(idx, unit)` — time span between first & last timestamp in chosen `unit` (`"years"`, `"months"`, `"weeks"`, `"days"`, `"hours"`, `"minutes"`, `"seconds"`, or `"bars"`).
- `bars_per_unit` — number of bars per chosen `unit` (e.g., for 1h bars and `unit="years"`, `bars_per_unit=8760`).

All rolling/recursive objects produce **NaN** during warm-up.

---

## Equity & PnL

### Equity Curve
Compounding from per-bar returns:

`E_t = E_0 * Π_{i=1..t} (1 + r_i)`

### Per-bar PnL
From equity:

`PnL_t = E_t - E_{t-1}`

Non-compounding mode (constant notional `N`):

`PnL_t = N * r_t`


---

## Return & Scaling Conventions

- **Nominal vs. excess:** when a risk-free rate `rf` is supplied, we use either:
  - **Difference in growth rates**: `CAGR − rf`, or
  - **Geometric excess**: `(1 + CAGR)/(1 + rf) − 1`.
- **Scaling:** dispersion metrics (volatility, Sharpe denominator) use `sqrt(bars_per_unit)`.

---

## Metrics

### ROI (Total Return)


`ROI = E_T / E_0 − 1`


### Excess ROI
Let `T = elapsed_from_index(idx, unit)`.

`ROI_excess = (E_T / E_0) / (1 + rf)^T − 1`


`rf` is per selected `unit`.

### CAGR (per unit)
Let `T = elapsed_from_index(idx, unit)`.

`CAGR = (E_T / E_0)^(1/T) − 1`



### Excess CAGR (geometric)

`ExcessCAGR = (1 + CAGR)/(1 + rf) − 1`


### Volatility (scaled)
For per-bar returns `r_t`:

`vol_unit = std(r_t) * sqrt(bars_per_unit) `
`vol_unit_excess = std(r_t − rf/bars_per_unit) * sqrt(bars_per_unit)`

### Sharpe Ratio

`x_t = r_t − rf/bars_per_unit`
`Sharpe = mean(x_t) / std(x_t) * sqrt(bars_per_unit)`


### Maximum Drawdown (MDD)
Let `RU_t = max(E_0, ..., E_t)`:

`DD_t = E_t / RU_t − 1`
`MDD = min_t DD_t # non-positive`


Also report timestamps of peak and trough.

### Calmar (Nominal)

`Calmar_nominal = CAGR_unit / |MDD|`

### Calmar (Excess, geometric)

`Calmar_excess = ExcessCAGR / |MDD|`

### Calmar (CAGR − Rf)

`Calmar_rf = (CAGR_unit − rf) / |MDD|`

### Hit Rate

`HitRate = mean( r_t > 0 )`


### Turnover
For a position series `pos_t` (e.g., `−1/0/+1` or continuous sizing):

`Turnover = Σ_t | pos_t − pos_{t−1} |`



---

## References



Wikipedia quick intros:
- Sharpe ratio — <https://en.wikipedia.org/wiki/Sharpe_ratio>  
- Drawdown — <https://en.wikipedia.org/wiki/Drawdown_(economics)>  
- Calmar ratio — <https://en.wikipedia.org/wiki/Calmar_ratio>  
- Rate of return — <https://en.wikipedia.org/wiki/Rate_of_return>  
- Standard deviation — <https://en.wikipedia.org/wiki/Standard_deviation>

---

## Usage

```python
import pandas as pd
from hedge.metrics import (
    equity_curve, pnl_from_equity, pnl_from_returns,
    roi, roi_excess, cagr, excess_cagr,
    volatility, sharpe, max_drawdown,
    calmar_nominal, calmar_excess, calmar_rf, calmar,
    hit_rate, turnover,
)

# Suppose we have per-bar returns and positions
rets = df["strategy_ret"]          # fractional per-bar returns
pos  = df["position"]              # -1/0/+1 or continuous
bars_per_year = 8760               # for 1h bars

# Equity & PnL
equity = equity_curve(rets, init_equity=1_000.0)
pnl    = pnl_from_equity(equity)   # or pnl_from_returns(rets, reinvest=False, fixed_notional=10_000)

# Core metrics
R_total   = roi(equity)
R_total_x = roi_excess(equity, rf=0.05, time_unit="years")

g_yr      = cagr(equity, time_unit="years")
g_yr_x    = excess_cagr(equity, rf_annual=0.05, time_unit="years")

vol_yr    = volatility(rets, bars_per_unit=bars_per_year)
shp_yr    = sharpe(rets, bars_per_unit=bars_per_year, rf=0.05)

mdd, t_peak, t_trough = max_drawdown(equity)

# Calmar variants
cal_nom   = calmar_nominal(equity)
cal_exc   = calmar_excess(equity, rf=0.05, time_unit="years")
cal_rf    = calmar_rf(equity, rf=0.05, time_unit="years")

# Convenience wrapper
cal_any   = calmar(equity, mode="rf", rf=0.05, time_unit="years")

# Other descriptive stats
hit = hit_rate(rets)
to  = turnover(pos)

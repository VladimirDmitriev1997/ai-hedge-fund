# Loss Functions & Utilities — Theory & Formulas

This module provides backend-agnostic (**NumPy / PyTorch**) losses and utilities for fitting portfolio strategies that output **weights** (including `CASH`). Everything is vectorized, causal-friendly, and compatible with autodiff.

## Contents
- [Conventions & Notation](#conventions--notation)
- [Portfolio Returns](#portfolio-returns)
- [Trading Costs (Turnover Model)](#trading-costs-turnover-model)
- [Wealth Curve](#wealth-curve)
- [Objectives (minimize)](#objectives-minimize)
  - [Negative Log-Wealth](#negative-log-wealth)
  - [Negative Sharpe (Excess)](#negative-sharpe-excess)
  - [Smooth Drawdown Surrogate](#smooth-drawdown-surrogate)
- [Regularizers](#regularizers)
- [End-to-End Objective](#end-to-end-objective)
- [Torch / NumPy Compatibility](#torch--numpy-compatibility)
- [Usage](#usage)
- [References](#references)

---

## Conventions & Notation

**Time-major shapes:**

- **Weights** \(W \in \mathbb{R}^{T\times M}\) — row `t` is the allocation **after** a rebalance, applied over \([t, t+1]\).  
  Rows should **sum to 1** including a `CASH` column.

- **Returns** \(R \in \mathbb{R}^{T\times M?}\) — per-asset *simple* returns over \([t, t+1]\).  
  If `R` **excludes** `CASH` (shape `(T, M-1)`), pass `rf` to inject the cash leg.

- **Risk-free** `rf` — per-period `CASH` return (scalar or `(T,)`).

Indexing is **causal**: decisions at time `t` apply to returns over `[t, t+1]`.

---

## Portfolio Returns

Given weights and asset returns (with `CASH` handled correctly):
\[
r^p_t \;=\; \sum_{j=1}^{M} w_{t,j}\, r_{t,j}.
\]

**Code:**  
`portfolio_returns(weights, asset_returns, rf=..., has_cash_in_returns=None, cash_col_idx=None)`

- If returns include `CASH`: use directly.  
- If returns exclude `CASH`: inject `rf` into the `CASH` column.

**Returns:** vector `(T,)` of per-period portfolio returns.

---

## Trading Costs (Turnover Model)

We model proportional fees/slippage from turnover:
\[
\text{cost}_t \;=\; \sum_{j=1}^{M}\, \big|w_{t,j}-w_{t-1,j}\big| \; \kappa_{t,j}, 
\qquad
\kappa_{t,j} \;=\; \frac{\text{fee\_bps}+\text{slippage\_bps}}{10^4}\cdot m_{t,j}.
\]

`m_{t,j}` is an optional per-asset multiplier (e.g., ATR, spread, venue tier).  
First-step turnover is computed vs. `initial_weights` (defaults to `0`).

**Code:**  
`costs_turnover(weights, fee_bps=..., slippage_bps=..., per_asset_multipliers=None, initial_weights=None)`

**Returns:** `(T,)` cost rates to subtract from \(r^p_t\).

---

## Wealth Curve

Self-financing wealth recursion with costs:
\[
E_{t+1} \;=\; E_t \,\big(1 + r^p_t - \text{cost}_t\big).
\]

**Code:**  
`wealth_curve(port_ret, costs=None, init_equity=1.0, clip_min=-0.999999)`

**Returns:** equity curve `(T,)`. Growth factors are clipped below `-1` for numeric safety.

---

## Objectives (minimize)

All objectives return a **scalar** suitable for minimization, and accept either **NumPy arrays** or **Torch tensors**.

### Negative Log-Wealth

Maximize terminal log-wealth (Kelly-like) by minimizing:
\[
\mathcal{L}_{\log W} \;=\; -\sum_{t=1}^{T}\log\!\big(1 + r^p_t - \text{cost}_t + \varepsilon\big).
\]

**Code:**  
`neg_log_wealth(port_ret, costs=None, eps=1e-12)`

---

### Negative Sharpe (Excess)

Sharpe of **net excess returns** \(x_t = r^p_t - \text{cost}_t - r^{f}_t\), annualized:
\[
\text{Sharpe} \;\approx\; \frac{\mathbb{E}[x]}{\text{Std}[x]}\sqrt{\text{annualizer}}
\quad\Rightarrow\quad
\mathcal{L}_{\text{Sharpe}} \;=\; -\,\text{Sharpe}.
\]

**Code:**  
`neg_sharpe(port_ret, costs=None, annualizer=..., rf_per_period=0.0, eps=1e-12)`

- `annualizer`: e.g., `365` (daily), `8760` (hourly).  
- `rf_per_period`: scalar or `(T,)` time series.

---

### Smooth Drawdown Surrogate

Path-aware penalty built from the equity curve \(E_t\). Let \(P_t=\max_{s\le t}E_s\).  
We penalize a **soft relative drawdown**:
\[
\mathcal{L}_{\text{DD}}
\;=\;
\frac{1}{T}\sum_{t=1}^{T}
\frac{1}{\tau}\,\text{softplus}\!\left(
\tau \cdot \frac{P_t - E_t}{P_t + \varepsilon}
\right).
\]

- Larger when drawdowns are **deeper/longer**.  
- `τ` controls smoothness: higher → closer to hard drawdown but gradients get sharper.

**Code:**  
`drawdown_surrogate(port_ret, costs=None, init_equity=1.0, tau=10.0, eps=1e-12)`

---

## Regularizers

Stabilize trading and control risk/complexity via simple **L1** penalties:
\[
\mathcal{R}
\;=\;
\lambda_{\text{to}}\;\mathbb{E}\!\Big[\sum_j | \Delta w_{t,j} |\Big]
\;+\;
\lambda_{\text{lev}}\;\mathbb{E}\!\Big[\sum_j | w_{t,j} |\Big].
\]

- **Turnover** discourages frequent rebalances (reduces fees/overfitting to noise).  
- **Leverage** discourages large absolute weights (box-ish control; works with shorts too).

**Code:**  
`regularizers(weights, l1_turnover=0.0, l1_leverage=0.0)`

---

## End-to-End Objective

One call that composes: **weights + returns → portfolio returns → costs → base loss → regularizers.**

```python
from losses import evaluate_objective, Objective

spec = Objective(
    name="neg_sharpe",        # or "neg_log_wealth", "drawdown"
    annualizer=8760,          # in case of annual target
    rf_per_period=0.0,
    weight_turnover=1e-3,
    weight_leverage=1e-3,
)

loss = evaluate_objective(
    weights=W,                # (T, M) includes CASH
    asset_returns=R,          # (T, M) or (T, M-1) if no CASH
    rf=rf_vec,                # used if R excludes CASH
    has_cash_in_returns=None,
    cash_col_idx=None,  
    fee_bps=5.0,
    slippage_bps=10.0,
    per_asset_multipliers=atr_or_spread_panel,
    initial_weights=w0,     
    spec=spec,
)


**What it does:**

- Builds portfolio returns \(r_t^p\) with CASH handled correctly.
- Computes cost rates from turnover (with fees/slippage).
- Applies the chosen base objective (log-wealth / Sharpe / drawdown).
- Adds regularizers (turnover/leverage) if requested.
- Returns a single scalar (NumPy float or Torch scalar tensor) to minimize.

---

## Torch / NumPy Compatibility

- Every function accepts NumPy or Torch inputs and returns the same backend type.
- If inputs are Torch tensors, all math runs through Torch for autodiff.
- Broadcasting rules for `rf`, `per_asset_multipliers`, etc., match NumPy/Torch semantics.

---

## Usage

### 1) Grid / black-box over strategy hyper-parameters

```python
# Produce weights W(theta) from your strategy on training data
W = strategy.predict_weights(df).weights.values  # (T, M) with CASH column
R = returns_panel.values                         # (T, M-1) risky, no CASH
rf = rf_per_period.values                        # (T,)

from losses import evaluate_objective, Objective

spec = Objective(name="neg_sharpe", annualizer=8760, rf_per_period=rf.mean())
loss = evaluate_objective(W, R, rf=rf, fee_bps=5, slippage_bps=10, spec=spec)
# minimize loss over theta via grid/search/Bayesian optimizer


### 2) Backprop with Torch (end-to-end differentiable)

import torch
from losses import evaluate_objective, Objective

W = torch.tensor(W_np, dtype=torch.float32, requires_grad=True)
R = torch.tensor(R_np, dtype=torch.float32)
rf = torch.tensor(rf_np, dtype=torch.float32)

spec = Objective(name="neg_log_wealth")
loss = evaluate_objective(W, R, rf=rf, spec=spec)
loss.backward()  # gradients wrt W (or wrt parameters that create W)

### 3) Drawdown-aware fitting with cost-aware penalties

from losses import evaluate_objective, Objective

spec = Objective(
    name="drawdown",
    init_equity=1.0,
    tau=8.0,
    weight_turnover=2e-3,
    weight_leverage=1e-3,
)
loss = evaluate_objective(W, R, rf=rf, fee_bps=3, slippage_bps=7, spec=spec)


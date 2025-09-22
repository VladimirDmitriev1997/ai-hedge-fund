# Feature Library (OHLCV) — Theory & Formulas

This project exposes a set of **pure, vectorized** features for OHLCV time series.  
They are designed for **research reproducibility** (deterministic, aligned, documented).

## Contents
- [Notation](#notation)
- [Returns](#returns)
- [Averages & Smoothing](#averages--smoothing)
  - [Simple Moving Average (SMA)](#simple-moving-average-sma)
  - [Exponential Moving Average (EMA)](#exponential-moving-average-ema)
  - [Weighted Moving Average (WMA)](#weighted-moving-average-wma)
- [Volatility & Dispersion](#volatility--dispersion)
  - [Rolling Volatility](#rolling-volatility)
  - [EWMA Volatility (RiskMetrics)](#ewma-volatility-riskmetrics)
  - [Z-score](#z-score)
- [Momentum & Oscillators](#momentum--oscillators)
  - [RSI (Wilder)](#relative-strength-index-rsi-wilder)
  - [MACD](#macd)
- [Bands & Ranges](#bands--ranges)
  - [Bollinger Bands](#bollinger-bands)
  - [True Range & ATR](#true-range--atr)
- [Rolling Extrema / Sums](#rolling-extrema--sums)
- [References](#references)
- [Usage](#usage)

---

## Notation

Let \( C_t \) be the **close** price at time \( t \) (index strictly increasing, UTC).  
Let \( H_t, L_t \) be high/low, and \( V_t \) volume.  
Window sizes are integers \( n \ge 1 \). Exponentially weighted objects use span \( s \ge 1 \) with
\[
\alpha = \frac{2}{s+1} \quad\text{(EMA smoothing factor)}.
\]
All rolling objects return **NaN** during warm-up periods.

---

## Returns

**Simple returns**
\[
r_t \;=\; \frac{C_t}{C_{t-1}} - 1.
\]

**Log returns**
\[
\ell_t \;=\; \log C_t \;-\; \log C_{t-1}.
\]

> Use log returns when aggregating across time; use simple returns for backtests that compound equity as \( E_{t+1} = E_t (1 + r_{t+1}) \).

---

## Averages & Smoothing

### Simple Moving Average (SMA)
\[
\operatorname{SMA}_t(n) \;=\; \frac{1}{n}\sum_{i=0}^{n-1} C_{t-i}.
\]

- Smooths noise uniformly.  
- Warm-up: first \( n-1 \) values are NaN.

**Code:** `sma(close, window=n)`

---

### Exponential Moving Average (EMA)

Define \( \alpha = \tfrac{2}{s+1} \) for span \( s \). Then
\[
\operatorname{EMA}_t(s) \;=\; \alpha\, C_t \;+\; (1-\alpha)\,\operatorname{EMA}_{t-1}(s), \qquad
\operatorname{EMA}_0(s) = C_0 \;\; \text{(or the first available mean)}.
\]

- Reacts faster than SMA (more weight to recent observations).  
- In pandas: `ewm(span=s, adjust=False)`.

**Code:** `ema(close, span=s)`

---

### Weighted Moving Average (WMA)

Linearly weighted over the window (recent points heavier):
\[
\operatorname{WMA}_t(n) \;=\; \frac{\sum_{i=1}^{n} i \cdot C_{t-n+i}}
{\sum_{i=1}^{n} i } \;=\; \frac{\sum_{i=1}^{n} i \cdot C_{t-n+i}}{n(n+1)/2}.
\]

**Code:** `wma(close, window=n)`

---

## Volatility & Dispersion

### Rolling Volatility

Population-like rolling standard deviation of a series \( x_t \) (often returns):
\[
\sigma_t(n) \;=\; 
\sqrt{ \frac{1}{n} \sum_{i=0}^{n-1} \big(x_{t-i} - \bar{x}_t\big)^2 },
\qquad
\bar{x}_t \;=\; \operatorname{SMA}_t(n).
\]

(If you use sample std, replace \( \tfrac{1}{n} \) by \( \tfrac{1}{n-1} \), i.e., \( \text{ddof}=1 \).)

**Code:** `rolling_vol(x, window=n, ddof=0)`

---

### EWMA Volatility (RiskMetrics)

Two equivalent presentations are common:

**(A) Centered EWMA (mean + variance)**
\[
\begin{aligned}
m_t \;&=\; (1-\alpha)\,m_{t-1} + \alpha\, x_t,\\
v_t \;&=\; (1-\alpha)\,v_{t-1} + \alpha\, (x_t - m_t)^2,\\
\sigma^{\text{EWMA}}_t \;&=\; \sqrt{v_t},
\end{aligned}
\]
with \( \alpha = 2/(s+1) \).

**(B) RiskMetrics variance recursion (zero-mean assumption)**
\[
\sigma_t^2 \;=\; \lambda\, \sigma_{t-1}^2 \;+\; (1-\lambda)\, x_t^2,
\]
where \( \lambda \approx 0.94 \) for daily data (original recommendation).  
Relation between forms: \( \lambda = 1 - \alpha \).

**Initialization:** set \( m_0 = x_0 \), \( v_0 = 0 \) (or sample estimates from the first window).

**Code:** `ewma_vol(x, span=s)`

---

### Z-score

\[
z_t(n) \;=\; \frac{x_t - \operatorname{SMA}_t(n)}{\operatorname{Std}_t(n)}.
\]

- If \( \operatorname{Std}_t(n)=0 \), define \( z_t=\text{NaN} \) to avoid infinities.  
- Useful for normalization and mean-reversion features.

**Code:** `zscore(x, window=n)`

---

## Momentum & Oscillators

### Relative Strength Index (RSI, Wilder)

Let \( \Delta_t = C_t - C_{t-1} \), gains \( G_t = \max(\Delta_t, 0) \), losses \( L_t = \max(-\Delta_t, 0) \).  
Wilder’s smoothing uses \( \alpha = 1/n \) with the **recursive** averages
\[
\begin{aligned}
\operatorname{AG}_t &= (1-\alpha)\,\operatorname{AG}_{t-1} + \alpha\, G_t,\\
\operatorname{AL}_t &= (1-\alpha)\,\operatorname{AL}_{t-1} + \alpha\, L_t,\\
\text{RS}_t &= \frac{\operatorname{AG}_t}{\operatorname{AL}_t},\qquad
\text{RSI}_t = 100 - \frac{100}{1+\text{RS}_t}.
\end{aligned}
\]

**Initialization:** compute AG/AL from the first \( n \) observations (simple averages), then apply the recursion.

**Code:** `rsi(close, window=14)`

---

### MACD

\[
\operatorname{MACD}_t \;=\; \operatorname{EMA}_t(s_{\text{fast}}) \;-\; \operatorname{EMA}_t(s_{\text{slow}}),\qquad
\text{signal}_t \;=\; \operatorname{EMA}_t( \operatorname{MACD}, s_{\text{sig}} ),\qquad
\text{hist}_t \;=\; \operatorname{MACD}_t \;-\; \text{signal}_t.
\]

Default spans: \( 12, 26, 9 \).  
**Initialization:** each EMA initialized as in the EMA section.

**Code:** `macd(close, fast_span=12, slow_span=26, signal_span=9)`  
**Returns:** DataFrame with columns `macd`, `signal`, `hist`.

---

## Bands & Ranges

### Bollinger Bands

\[
\begin{aligned}
\text{mid}_t &= \operatorname{SMA}_t(n), \quad
\text{sd}_t = \operatorname{Std}_t(n),\\
\text{upper}_t &= \text{mid}_t + k\,\text{sd}_t, \quad
\text{lower}_t = \text{mid}_t - k\,\text{sd}_t.
\end{aligned}
\]

Auxiliary measures:
\[
\text{bandwidth}_t = \frac{\text{upper}_t - \text{lower}_t}{\text{mid}_t},
\qquad
\%b_t = \frac{C_t - \text{lower}_t}{\text{upper}_t - \text{lower}_t}.
\]

Typical \( n=20 \), \( k=2 \).

**Code:** `bollinger_bands(close, window=20, num_std=2.0)`

---

### True Range & ATR

**True Range (per bar):**
\[
\operatorname{TR}_t \;=\; \max\!\Big( 
H_t - L_t,\; \big|H_t - C_{t-1}\big|,\; \big|L_t - C_{t-1}\big|
\Big).
\]

**Average True Range (Wilder, \( \alpha=1/n \)):**
\[
\operatorname{ATR}_t(n) \;=\; (1-\alpha)\,\operatorname{ATR}_{t-1}(n) \;+\; \alpha\, \operatorname{TR}_t,
\qquad \alpha = \frac{1}{n}.
\]

**Initialization:** \( \operatorname{ATR}_{n} = \frac{1}{n}\sum_{i=1}^{n}\operatorname{TR}_i \).

**Code:** `true_range(high, low, close)` and `atr(high, low, close, window=14)`

---

## Rolling Extrema / Sums

\[
\begin{aligned}
\operatorname{RollMin}_t(n) &= \min\{x_{t-i}\}_{i=0}^{n-1},\\
\operatorname{RollMax}_t(n) &= \max\{x_{t-i}\}_{i=0}^{n-1},\\
\operatorname{RollSum}_t(n) &= \sum_{i=0}^{n-1} x_{t-i}.
\end{aligned}
\]

**Code:** `rolling_min(x,n)`, `rolling_max(x,n)`, `rolling_sum(x,n)`

---

## References

**General & mathy**
- RiskMetrics Group (1996). *RiskMetrics™ — Technical Document* (EWMA volatility).
- Grinold, R., Kahn, R. (2000). *Active Portfolio Management.*
- López de Prado, M. (2018). *Advances in Financial Machine Learning.*

**Wikipedia (quick intros)**
- Moving average — <https://en.wikipedia.org/wiki/Moving_average>  
- Exponential smoothing / EMA — <https://en.wikipedia.org/wiki/Exponential_smoothing>  
- RSI — <https://en.wikipedia.org/wiki/Relative_strength_index>  
- MACD — <https://en.wikipedia.org/wiki/MACD>  
- Bollinger Bands — <https://en.wikipedia.org/wiki/Bollinger_Bands>  
- Average True Range — <https://en.wikipedia.org/wiki/Average_true_range>  
- Volatility — <https://en.wikipedia.org/wiki/Volatility_(finance)>  
- Standard score (Z-score) — <https://en.wikipedia.org/wiki/Standard_score>

---

## Usage

```python
import pandas as pd
from hedge.features import (
    sma, ema, wma,
    simple_returns, log_returns,
    rolling_vol, ewma_vol, zscore,
    rsi, macd, bollinger_bands, atr,
    rolling_min, rolling_max, rolling_sum
)

# df has columns: open, high, low, close, volume; DatetimeIndex in UTC
close = df["close"]
high, low = df["high"], df["low"]

feat = pd.DataFrame(index=df.index)
feat["ret1"]    = simple_returns(close)
feat["sma20"]   = sma(close, 20)
feat["ema50"]   = ema(close, 50)
feat["wma10"]   = wma(close, 10)
feat["vol20"]   = rolling_vol(feat["ret1"], 20)
feat["ewvol64"] = ewma_vol(feat["ret1"], 64)
feat["z20"]     = zscore(close, 20)
feat["rsi14"]   = rsi(close, 14)

macd_df = macd(close)  # macd, signal, hist
bb      = bollinger_bands(close, 20, 2.0)

feat = feat.join(macd_df).join(bb)
feat["atr14"] = atr(high, low, close, 14)

# Use these features in your signal(s) or ML pipeline


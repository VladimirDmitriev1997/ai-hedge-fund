# Data Guide (OHLCV)

This project uses time-bar data in **OHLCV** format:
- **O**pen — first traded price in the bar’s time window  
- **H**igh — max trade price within the window  
- **L**ow — min trade price within the window  
- **C**lose — last traded price at the end of the window  
- **V**olume — sum of traded units in the window

All timestamps are stored in **UTC** and sorted strictly increasing. One row = one time bar.

### Resampling
If your raw data are at a finer resolution (e.g., 1-minute) and you need coarser bars (e.g., 1-hour):  
use **proper OHLC aggregation** (open=first, high=max, low=min, close=last, volume=sum) with right-closed bins (each bar covers `(t_prev, t]`). See `hedge.data.resample_ohlcv()`.

### Recommended naming:
- Single pair/hourly: `BTCUSDT_1h.csv`
- Single pair/4h:     `BTCUSDT_4h.csv`
- Raw minute data:    `BTCUSDT_1m.csv` (then resample in code)

Large raw datasets should **not** be committed to git; keep only small samples or resampled subsets for examples.

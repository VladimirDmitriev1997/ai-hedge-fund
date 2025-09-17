"""
Data loading and validation OHLCV time series.

We assume CSV with columns:
timestamp, open, high, low, close, volume
- timestamp should be parseable to UTC-aware pandas.Datetime
- rows must be sorted by time and duplicates removed
- gaps are allowed
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import re
import time
from datetime import datetime, timezone
import requests


_BINANCE_BASE = "https://api.binance.com"


@dataclass(frozen=True)
class DataScheme:
    """Interface for reading an OHLCV CSV."""

    path: Path | str
    tz: str = "UTC"
    parse_unit: Literal["ns", "ms", "s", "auto"] = "auto"
    expected_cols: tuple[str, ...] = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )


def load_ohlcv_csv(scheme: DataScheme) -> pd.DataFrame:
    """
    Load an OHLCV CSV into UTC-indexed DataFrame.

    Returns
    -------
    DataFrame with columns: open, high, low, close, volume
    and a DatetimeIndex in UTC, strictly increasing, no duplicates.

    Notes
    -----
    Use resample_ohlcv() below to change timeframe.
    """
    path = Path(scheme.path)

    df = pd.read_csv(path)

    # Basic schema sanity
    missing = set(scheme.expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing columns: {sorted(missing)}")

    ts = df["timestamp"]

    # If timestamps are integers (epoch), we need to get unit (ns/ms/s). Heuristics for unit definition see in _infer_epoch_unit():
    if np.issubdtype(ts.dtype, np.number):
        unit = (
            _infer_epoch_unit(ts) if scheme.parse_unit == "auto" else scheme.parse_unit
        )
        dt = pd.to_datetime(ts.astype("int64"), unit=unit, utc=True)
    else:
        # timestamps
        dt = pd.to_datetime(ts, utc=True, errors="coerce")

    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"Failed to parse {bad} timestamps in {path}.")

    # normalize to UTC timezone - this timezone is expected
    dt = dt.tz_convert(scheme.tz) if str(dt.dt.tz) != scheme.tz else dt

    num_cols = ["open", "high", "low", "close", "volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df[num_cols].copy()
    out.index = pd.DatetimeIndex(dt, name="timestamp")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Prices non-negative, highs >= lows, volume >= 0
    if (out[["open", "high", "low", "close"]] < 0).any().any():
        raise ValueError("Negative price found — corrupted CSV?")
    if (out["high"] < out["low"]).any():
        raise ValueError("Row with high < low — corrupted OHLC?")

    # Drop rows with NaNs that we created during coercion
    if out.isna().any().any():

        out = out.dropna()

    return out


def resample_ohlcv(
    df: pd.DataFrame,
    rule: str,
    how: Literal["ohlc", "close"] = "ohlc",
    label: Literal["left", "right"] = "right",
    closed: Literal["left", "right"] = "right",
) -> pd.DataFrame:
    """
    Resample an OHLCV DataFrame to a new timeframe (e.g., '1h' -> '4h', '1D').

    Parameters
    ----------
    df : DataFrame
        Must have columns open, high, low, close, volume and a tz-aware DatetimeIndex.
    rule : str
        Pandas offset alias, e.g., '15min', '1h', '4h', '1D', '1W'.
    how : {'ohlc','close'}
        'ohlc'  -> proper OHLC aggregation
        'close' -> just reindex to the new period taking last close; volume summed
    label, closed : {'left','right'}
        Control how bins are labeled and which edge is inclusive.
        For crypto (24/7) (t_prev, t].

    Returns
    -------
    Resampled DataFrame with same columns.
    """
    if how == "ohlc":
        o = df["open"].resample(rule, label=label, closed=closed).first()
        h = df["high"].resample(rule, label=label, closed=closed).max()
        l = df["low"].resample(rule, label=label, closed=closed).min()
        c = df["close"].resample(rule, label=label, closed=closed).last()
        v = df["volume"].resample(rule, label=label, closed=closed).sum()
        out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    elif how == "close":
        c = df["close"].resample(rule, label=label, closed=closed).last()
        v = df["volume"].resample(rule, label=label, closed=closed).sum()
        out = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": v})
    else:
        raise ValueError("how must be 'ohlc' or 'close'")

    return out.dropna()


# ---- Helpers --------------------------------------------------------------------


def _infer_epoch_unit(ts: pd.Series) -> Literal["ns", "ms", "s"]:
    """
    Heuristic for integer epoch timestamps:
    - If values look like 13 digits -> milliseconds
    - 10 digits -> seconds
    - 19 digits -> nanoseconds
    """
    # take a first non-NaN
    x = int(pd.Series(ts).dropna().iloc[0])
    n = len(str(abs(x)))
    if n >= 19:
        return "ns"
    if n >= 13:
        return "ms"
    return "s"


def _to_ms(dt_like: str | int | float | datetime | None) -> Optional[int]:
    """
    Convert many date-like inputs to milliseconds.

    Parameters
    ----------
    dt_like: accepts
      - None
      - datetime (naive or tz-aware)
      - epoch as int/float (s/ms/ns)
      - string: ISO-like "YYYY-MM-DD[ HH:MM[:SS]]", or numeric epoch string

    Returns
    ----------
    Input converted to milliseconds
    """
    if dt_like is None:
        return None

    # datetime to ms
    if isinstance(dt_like, datetime):
        if dt_like.tzinfo is None:
            dt_like = dt_like.replace(tzinfo=timezone.utc)
        return int(dt_like.timestamp() * 1000)

    # numpy scalars (e.g., np.int64) - treat like Python numbers
    if isinstance(dt_like, (np.integer, np.floating)):
        dt_like = dt_like.item()

    # numeric epoch
    if isinstance(dt_like, (int, float)):
        x = int(round(dt_like))
        unit = _infer_epoch_unit(pd.Series([x]))
        ts = pd.to_datetime(x, unit=unit, utc=True)
        return int(ts.to_datetime64().astype("datetime64[ms]").astype("int64"))

    # tring:
    if isinstance(dt_like, str):
        s = dt_like.strip()

        s_compact = re.sub(r"[ _]", "", s)
        if re.fullmatch(r"[+-]?\d{9,}", s_compact):
            x = int(s_compact)
            unit = _infer_epoch_unit(pd.Series([x]))
            ts = pd.to_datetime(x, unit=unit, utc=True)
            return int(ts.to_datetime64().astype("datetime64[ms]").astype("int64"))

        # ISO-like calendar string
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Cannot parse datetime string: {dt_like!r}")
        return int(ts.to_datetime64().astype("datetime64[ms]").astype("int64"))

    # Fallback: unsupported type
    raise TypeError(f"Unsupported datetime-like type: {type(dt_like)!r}")


def download_binance_ohlcv_csv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str | int | float | datetime | None = "2022-01-01",
    end: str | int | float | datetime | None = None,
    out_path: str | Path = "data/BTCUSDT_1h.csv",
    limit_per_call: int = 1000,
    request_pause_sec: float = 0.25,
) -> Path:
    """
    Download OHLCV from Binance API and save as CSV with columns:
    timestamp,open,high,low,close,volume

    Parameters
    ----------
    symbol : for example 'BTCUSDT'
    interval : '1m','5m','15m','1h','4h','1d'
    start, end : inclusive time range (string like '2022-01-01', epoch s/ms, or datetime)
    out_path : destination CSV path
    limit_per_call : Binance max items per request
    request_pause_sec : small sleep to be polite and avoid 429s

    Notes
    -----
    - We keep only OHLCV and set timestamp = closeTime (UTC).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end is not None else _to_ms(datetime.now(timezone.utc))

    url = f"{_BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit_per_call, 1000),
    }

    rows: list[tuple[int, str, str, str, str, str]] = []
    cur = start_ms
    while True:
        q = dict(params)
        q["startTime"] = cur
        q["endTime"] = end_ms
        resp = requests.get(url, params=q, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance API error {resp.status_code}: {resp.text}")

        data = resp.json()
        if not data:
            break

        # Each kline: [ openTime, open, high, low, close, volume, closeTime, ... ]
        for k in data:
            close_time_ms = int(k[6])
            # stop if we exceeded end
            if close_time_ms > end_ms:
                break
            rows.append(
                (
                    close_time_ms,
                    k[1],  # open
                    k[2],  # high
                    k[3],  # low
                    k[4],  # close
                    k[5],  # volume
                )
            )

        # If we received less than the limit, we reached the end
        if len(data) < params["limit"]:
            break

        # Otherwise advance start to the last close_time
        last_close_ms = int(data[-1][6])
        if last_close_ms >= end_ms:
            break
        cur = last_close_ms + 1

        time.sleep(request_pause_sec)

    if not rows:
        raise RuntimeError("No data returned for the requested range.")

    # Build DataFrame and write CSV compatible with our loader
    df = pd.DataFrame(
        rows, columns=["close_time_ms", "open", "high", "low", "close", "volume"]
    )
    # Convert close_time_ms to ISO UTC strings
    ts = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
    df_out = pd.DataFrame(
        {
            "timestamp": ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    )
    df_out.to_csv(out_path, index=False)
    return out_path

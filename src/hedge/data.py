"""
Data loading and validation for OHLCV time series, plus portfolio dataset builder.

Assumptions
-----------
CSV schema: timestamp, open, high, low, close, volume
- timestamp must parse to a tz-aware DatetimeIndex in UTC (or convertible).
- rows are sorted strictly increasing by time; duplicates are removed.
- gaps are allowed (no synthetic fill).

New utilities
-------------
- fetch_binance_ohlcv_df(...) : pull OHLCV for one symbol/interval to a DataFrame.
- build_and_save_portfolio_dataset(...):
    * fetch multiple symbols,
    * select per-symbol columns (e.g., close / ohlc / volume),
    * outer-join into a single MultiIndex-column DataFrame: (symbol, field),
    * save to CSV: portfolio_<code>.csv,
    * append a line with metadata to portfolio_catalog.csv in the same directory.

All numeric columns are float64. Index is tz-aware UTC, strictly increasing.

Notes
-----
- Resampling helper uses right-closed bins (t_prev, t] which is typical for crypto.
- For Binance klines, we set timestamp = kline closeTime (UTC).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import re
import time
from datetime import datetime, timezone
import requests


_BINANCE_BASE = "https://api.binance.com"

ALLOWED_FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")


# ------------------------------------------------------------------------------
# Core CSV loader
# ------------------------------------------------------------------------------


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
    DataFrame
        Columns: open, high, low, close, volume (float64)
        Index: tz-aware DatetimeIndex in UTC, strictly increasing, no duplicates.

    Notes
    -----
    Use `resample_ohlcv()` to change timeframe.
    """
    path = Path(scheme.path)
    df = pd.read_csv(path)

    # Schema check
    missing = set(scheme.expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing columns: {sorted(missing)}")

    ts = df["timestamp"]

    # Parse timestamps
    if pd.api.types.is_numeric_dtype(ts):
        unit = (
            _infer_epoch_unit(ts) if scheme.parse_unit == "auto" else scheme.parse_unit
        )
        dt = pd.to_datetime(ts.astype("int64"), unit=unit, utc=True)
    else:
        dt = pd.to_datetime(ts, utc=True, errors="coerce")

    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"Failed to parse {bad} timestamps in {path}.")

    # Convert to desired timezone (default UTC)
    if isinstance(dt, pd.DatetimeIndex):
        if scheme.tz and (str(dt.tz) != scheme.tz):
            dt = dt.tz_convert(scheme.tz)
    else:
        # Coerce to DatetimeIndex
        dt = pd.DatetimeIndex(dt).tz_convert(scheme.tz)

    # Numeric coercion → float64
    num_cols = ["open", "high", "low", "close", "volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    out = df[num_cols].copy()
    out.index = pd.DatetimeIndex(dt, name="timestamp")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Basic sanity checks
    if (out[["open", "high", "low", "close"]] < 0).any().any():
        raise ValueError("Negative price found — corrupted CSV?")
    if (out["high"] < out["low"]).any():
        raise ValueError("Row with high < low — corrupted OHLC?")
    if (out["volume"] < 0).any():
        raise ValueError("Negative volume found — corrupted CSV?")

    # Drop rows with NaNs created during coercion
    out = out.dropna(how="any")

    # Final dtype normalization
    out = out.astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        }
    )
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
        Columns open, high, low, close, volume; tz-aware DatetimeIndex.
    rule : str
        Pandas offset alias, e.g., '15min', '1h', '4h', '1D', '1W'.
    how : {'ohlc','close'}
        'ohlc'  -> proper OHLC aggregation
        'close' -> reindex to new period taking last close; volume summed
    label, closed : {'left','right'}
        For crypto (24/7) we typically use right-closed bins: (t_prev, t].

    Returns
    -------
    DataFrame
        Resampled with same columns; float64 dtypes.
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

    out = out.dropna(how="any")
    return out.astype("float64")


# ------------------------------------------------------------------------------
# Binance helpers
# ------------------------------------------------------------------------------


def _infer_epoch_unit(ts: pd.Series) -> Literal["ns", "ms", "s"]:
    """
    Heuristic for integer epoch timestamps:
    - 19+ digits -> nanoseconds
    - 13-18 digits -> milliseconds
    - else -> seconds
    """
    x = int(pd.Series(ts).dropna().iloc[0])
    n = len(str(abs(x)))
    if n >= 19:
        return "ns"
    if n >= 13:
        return "ms"
    return "s"


def _to_ms(dt_like: str | int | float | datetime | None) -> Optional[int]:
    """
    Convert a variety of date-like inputs to milliseconds since epoch (UTC).

    Accepts
    -------
    - None
    - datetime (naive or tz-aware)
    - epoch as int/float (s/ms/ns)
    - string: ISO-like "YYYY-MM-DD[ HH:MM[:SS]]", or numeric epoch string
    """
    if dt_like is None:
        return None

    # datetime → ms
    if isinstance(dt_like, datetime):
        if dt_like.tzinfo is None:
            dt_like = dt_like.replace(tzinfo=timezone.utc)
        return int(dt_like.timestamp() * 1000)

    # numpy scalars (e.g., np.int64)
    if isinstance(dt_like, (np.integer, np.floating)):
        dt_like = dt_like.item()

    # numeric epoch
    if isinstance(dt_like, (int, float)):
        x = int(round(dt_like))
        unit = _infer_epoch_unit(pd.Series([x]))
        ts = pd.to_datetime(x, unit=unit, utc=True)
        return int(ts.to_datetime64().astype("datetime64[ms]").astype("int64"))

    # string
    if isinstance(dt_like, str):
        s = dt_like.strip()
        s_compact = re.sub(r"[ _]", "", s)
        if re.fullmatch(r"[+-]?\d{9,}", s_compact):
            x = int(s_compact)
            unit = _infer_epoch_unit(pd.Series([x]))
            ts = pd.to_datetime(x, unit=unit, utc=True)
            return int(ts.to_datetime64().astype("datetime64[ms]").astype("int64"))
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Cannot parse datetime string: {dt_like!r}")
        return int(ts.to_datetime64().astype("datetime64[ms]").astype("int64"))

    raise TypeError(f"Unsupported datetime-like type: {type(dt_like)!r}")


def fetch_binance_ohlcv_df(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str | int | float | datetime | None = "2022-01-01",
    end: str | int | float | datetime | None = None,
    *,
    limit_per_call: int = 1000,
    request_pause_sec: float = 0.25,
) -> pd.DataFrame:
    """
    Download OHLCV from Binance API into a DataFrame with UTC DatetimeIndex.

    Parameters
    ----------
    symbol : str
        e.g., 'BTCUSDT'
    interval : str
        One of Binance klines intervals: '1m','5m','15m','1h','4h','1d', etc.
    start, end : Any
        Inclusive time range; string date, epoch s/ms/ns, or datetime.
        If `end` is None, uses now (UTC).
    limit_per_call : int
        Max klines per request (≤ 1000).
    request_pause_sec : float
        Sleep between requests to avoid 429s.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume (float64);
        Index: timestamp at kline closeTime (UTC), strictly increasing.
    """
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end is not None else _to_ms(datetime.now(timezone.utc))

    url = f"{_BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(int(limit_per_call), 1000),
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
            if close_time_ms > end_ms:  # guard overshoot
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

        # If fewer than limit, we're done
        if len(data) < params["limit"]:
            break

        last_close_ms = int(data[-1][6])
        if last_close_ms >= end_ms:
            break
        cur = last_close_ms + 1

        time.sleep(request_pause_sec)

    if not rows:
        raise RuntimeError("No data returned for the requested range.")

    df = pd.DataFrame(
        rows, columns=["close_time_ms", "open", "high", "low", "close", "volume"]
    )
    ts = pd.to_datetime(df["close_time_ms"].astype("int64"), unit="ms", utc=True)
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df["open"], errors="coerce").astype("float64"),
            "high": pd.to_numeric(df["high"], errors="coerce").astype("float64"),
            "low": pd.to_numeric(df["low"], errors="coerce").astype("float64"),
            "close": pd.to_numeric(df["close"], errors="coerce").astype("float64"),
            "volume": pd.to_numeric(df["volume"], errors="coerce").astype("float64"),
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    ).sort_index()

    # Sanity checks
    out = out[~out.index.duplicated(keep="last")]
    if (out[["open", "high", "low", "close"]] < 0).any().any():
        raise ValueError(f"{symbol}: negative price found.")
    if (out["high"] < out["low"]).any():
        raise ValueError(f"{symbol}: high < low encountered.")
    if (out["volume"] < 0).any():
        raise ValueError(f"{symbol}: negative volume found.")

    return out


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
    Convenience wrapper: fetch Binance OHLCV and save as CSV with columns:
    timestamp,open,high,low,close,volume (timestamp in ISO UTC).

    Notes
    -----
    - Timestamp equals kline closeTime.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_binance_ohlcv_df(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        limit_per_call=limit_per_call,
        request_pause_sec=request_pause_sec,
    )
    df_out = df.copy()
    df_out = df_out.reset_index()
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df_out[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(
        out_path, index=False
    )
    return out_path


# ------------------------------------------------------------------------------
# Portfolio dataset builder + catalog logging
# ------------------------------------------------------------------------------


def build_and_save_portfolio_dataset(
    code: str,
    symbols_fields: Mapping[str, Sequence[str]],
    *,
    interval: str = "1h",
    start: str | int | float | datetime | None = "2022-01-01",
    end: str | int | float | datetime | None = None,
    out_dir: str | Path = "data",
    limit_per_call: int = 1000,
    request_pause_sec: float = 0.25,
    join_how: Literal["outer", "inner"] = "outer",
) -> Path:
    """
    Fetch multiple symbols and assemble a single MultiIndex-column DataFrame.

    Parameters
    ----------
    code : str
        Portfolio code used for file naming, e.g., "crypto_1h_2022".
    symbols_fields : Mapping[str, Sequence[str]]
        Dict of SYMBOL -> fields to keep (subset of {'open','high','low','close','volume'}).
        Example: {'BTCUSDT': ['close','volume'], 'ETHUSDT': ['close']}
    interval : str
        Binance kline interval ('1m','5m','15m','1h','4h','1d', ...).
    start, end : Any
        Inclusive time range (string date, epoch s/ms/ns, or datetime).
    out_dir : str | Path
        Output directory. Two files are produced here:
          - portfolio_<code>.csv
          - portfolio_catalog.csv  (append-only metadata log)
    limit_per_call, request_pause_sec : API pacing parameters.
    join_how : {'outer','inner'}
        Index join method across symbols (default outer to keep union of bars).

    Returns
    -------
    Path
        Path to the saved CSV dataset: <out_dir>/portfolio_<code>.csv

    Side effects
    ------------
    Appends metadata to <out_dir>/portfolio_catalog.csv.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_clean = re.sub(r"[^A-Za-z0-9_\-\.]", "_", code).strip("_")
    out_path = out_dir / f"portfolio_{code_clean}.csv"
    catalog_path = out_dir / "portfolio_catalog.csv"

    # Validate fields spec
    for sym, fields in symbols_fields.items():
        bad = [f for f in fields if f not in ALLOWED_FIELDS]
        if bad:
            raise ValueError(f"{sym}: unsupported fields requested: {bad}")

    # Fetch each symbol, select fields, rename to MultiIndex (sym, field)
    frames: list[pd.DataFrame] = []
    for sym, fields in symbols_fields.items():
        df = fetch_binance_ohlcv_df(
            symbol=sym,
            interval=interval,
            start=start,
            end=end,
            limit_per_call=limit_per_call,
            request_pause_sec=request_pause_sec,
        )
        # Select requested fields
        sub = df.loc[:, list(fields)].copy()
        # MultiIndex columns (symbol, field)
        sub.columns = pd.MultiIndex.from_product(
            [[sym.upper()], list(fields)], names=["symbol", "field"]
        )
        frames.append(sub)

    # Join along time
    if not frames:
        raise ValueError("symbols_fields is empty.")
    df_all = frames[0]
    for f in frames[1:]:
        df_all = df_all.join(f, how=join_how)

    # Ensure strictly increasing, tz-aware UTC index, float64 dtypes
    df_all = df_all.sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="last")]
    if not isinstance(df_all.index, pd.DatetimeIndex) or df_all.index.tz is None:
        # Should not happen (we fetch in UTC), but enforce for consistency
        df_all.index = pd.DatetimeIndex(df_all.index, tz="UTC", name="timestamp")

    # Save CSV with index as ISO UTC string
    df_out = df_all.copy()
    df_out = df_out.reset_index()
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df_out.to_csv(out_path, index=False)

    # Append to catalog
    _append_portfolio_catalog(
        catalog_path=catalog_path,
        code=code_clean,
        interval=interval,
        start=start,
        end=end,
        symbols_fields=symbols_fields,
        dataset_path=str(out_path),
    )

    return out_path


def _append_portfolio_catalog(
    catalog_path: Path,
    *,
    code: str,
    interval: str,
    start: str | int | float | datetime | None,
    end: str | int | float | datetime | None,
    symbols_fields: Mapping[str, Sequence[str]],
    dataset_path: str,
) -> None:
    """
    Append a single line of metadata to the portfolio catalog CSV.

    Columns
    -------
    created_utc, code, interval, start, end, symbols, fields_per_symbol, dataset_path

    - `symbols` is a semicolon-separated list like: "BTCUSDT;ETHUSDT"
    - `fields_per_symbol` is a semicolon-separated list mirroring symbols, e.g.:
         "close|volume;close"
    """
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    symbols = [s.upper() for s in symbols_fields.keys()]
    fields_repr = ["|".join(map(str, symbols_fields[s])) for s in symbols_fields]

    row = {
        "created_utc": created,
        "code": code,
        "interval": interval,
        "start": _repr_time_for_catalog(start),
        "end": _repr_time_for_catalog(end),
        "symbols": ";".join(symbols),
        "fields_per_symbol": ";".join(fields_repr),
        "dataset_path": dataset_path,
    }

    if catalog_path.exists():
        cat = pd.read_csv(catalog_path)
        cat = pd.concat([cat, pd.DataFrame([row])], ignore_index=True)
    else:
        cat = pd.DataFrame([row])
    cat.to_csv(catalog_path, index=False)


def _repr_time_for_catalog(x: str | int | float | datetime | None) -> str:
    """Human-friendly ISO-ish representation for the catalog."""
    if x is None:
        return ""
    if isinstance(x, datetime):
        if x.tzinfo is None:
            x = x.replace(tzinfo=timezone.utc)
        return x.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Try epoch-like or parsable string
    try:
        ms = _to_ms(x)  # type: ignore[arg-type]
        ts = pd.to_datetime(ms, unit="ms", utc=True)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(x)


def load_portfolio_csv(
    path: str | Path,
    *,
    tz: str = "UTC",
    fields_required: Sequence[str] | None = ("close",),
    flat_sep: str = "_",
) -> pd.DataFrame:
    """
    Load a multi-asset 'portfolio_<code>.csv' back into a MultiIndex-column DataFrame.

    Supports two header styles:
      1) MultiIndex header rows (what build_and_save_portfolio_dataset writes):
           timestamp  BTCUSDT                   ETHUSDT
                      close  volume             close
         ...rows...
         -> columns MultiIndex: (symbol, field) with names ['symbol','field']
      2) Flat columns like: timestamp, BTCUSDT_close, ETHUSDT_close, ...
         -> will be split by `flat_sep` into (symbol, field) if field∈ALLOWED_FIELDS.

    Parameters
    ----------
    path : str | Path
        CSV path (produced by build_and_save_portfolio_dataset).
    tz : str, default 'UTC'
        Target timezone for DatetimeIndex.
    fields_required : sequence[str] or None
        If provided, validate each symbol has these fields (e.g., ('close',)).
    flat_sep : str, default '_'
        Separator for parsing fallback flat columns 'SYMBOL_field'.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns (symbol, field), float64 dtypes, tz-aware UTC index.
    """
    path = Path(path)

    # First try: MultiIndex header
    try:
        df = pd.read_csv(path, header=[0, 1])
        # detect whether we really got a 2-level header (excluding the timestamp col)
        cols = df.columns
        is_multi = isinstance(cols, pd.MultiIndex)
        if is_multi:
            # timestamp column could be ('timestamp','') or ('timestamp','timestamp') depending on pandas
            # Find any column whose top level equals 'timestamp'
            ts_candidates = [
                c
                for c in cols
                if (isinstance(c, tuple) and str(c[0]).lower() == "timestamp")
            ]
            if ts_candidates:
                ts_col = ts_candidates[0]
                ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                if ts.isna().any():
                    raise ValueError(
                        "Failed to parse timestamps in portfolio CSV (multiheader)."
                    )
                X = df.drop(columns=[ts_col])
                X.columns = pd.MultiIndex.from_tuples(
                    [(str(a).upper(), str(b)) for (a, b) in X.columns],
                    names=["symbol", "field"],
                )
                X.index = pd.DatetimeIndex(ts, name="timestamp")
                # Normalize
                X = X.sort_index()
                X = X[~X.index.duplicated(keep="last")]
                if tz and str(X.index.tz) != tz:
                    X.index = X.index.tz_convert(tz)

                # Dtypes
                for c in X.columns:
                    X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
                X = X.dropna(how="any")

                # Optional field validation
                if fields_required:
                    req = set(fields_required)
                    by_sym = X.columns.get_level_values("symbol").unique()
                    for sym in by_sym:
                        have = set(X.loc[:, sym].columns)
                        missing = sorted(list(req - have))
                        if missing:
                            raise KeyError(f"{sym}: missing required fields {missing}")
                return X
    except Exception:
        # fall through to flat header parser
        pass

    # Fallback: flat header like 'BTCUSDT_close'
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Portfolio CSV must include a 'timestamp' column.")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Failed to parse timestamps in portfolio CSV.")

    # Build MultiIndex columns by splitting flat names
    cols = [c for c in df.columns if c != "timestamp"]
    tuples: list[tuple[str, str]] = []
    bad: list[str] = []
    for c in cols:
        parts = str(c).rsplit(flat_sep, 1)
        if len(parts) == 2 and parts[1] in ALLOWED_FIELDS:
            tuples.append((parts[0].upper(), parts[1]))
        else:
            bad.append(c)
    if not tuples:
        raise ValueError(
            "Could not detect MultiIndex columns. The CSV does not look like a portfolio export."
        )
    if bad:
        # non-fatal: we ignore unknown columns
        cols_ok = [c for c in cols if c not in bad]
    else:
        cols_ok = cols

    X = df[cols_ok].copy()
    X.columns = pd.MultiIndex.from_tuples(tuples, names=["symbol", "field"])
    X.index = pd.DatetimeIndex(ts, name="timestamp")

    # Normalize & dtypes
    X = X.sort_index()
    X = X[~X.index.duplicated(keep="last")]
    if tz and str(X.index.tz) != tz:
        X.index = X.index.tz_convert(tz)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    X = X.dropna(how="any")

    if fields_required:
        req = set(fields_required)
        by_sym = X.columns.get_level_values("symbol").unique()
        for sym in by_sym:
            have = set(X.loc[:, sym].columns)
            missing = sorted(list(req - have))
            if missing:
                raise KeyError(f"{sym}: missing required fields {missing}")

    return X

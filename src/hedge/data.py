"""
Data loading and validation for OHLCV time series, plus dataset/portfolio builders.

Assumptions
-----------
CSV schema: timestamp, open, high, low, close, volume
- timestamp parses to a tz-aware DatetimeIndex in UTC (or convertible).
- rows are strictly increasing by time; duplicates are removed.
- gaps are allowed (no synthetic fill).

New utilities
-------------
Single-asset catalog with numeric codes:
- ensure_ohlcv_dataset(symbol, interval, start, end, data_dir) -> (code, path)
  * Fetch OHLCV via Binance, save to data/<code>.csv, and register/lookup in
    data/dataset_catalog.csv. Code is a monotonically increasing integer.
- load_ohlcv_by_code(code, data_dir) -> DataFrame
- load_dataset_catalog(data_dir) -> DataFrame

Multi-asset helper (unchanged):
- build_and_save_portfolio_dataset(...) + load_portfolio_csv(...)

All numeric columns are float64. Index is tz-aware UTC, strictly increasing.
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

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

_BINANCE_BASE = "https://api.binance.com"

ALLOWED_FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")

DATASET_CATALOG_NAME = "dataset_catalog.csv"  # single-asset numeric-code catalog
PORTFOLIO_CATALOG_NAME = "portfolio_catalog.csv"  # multi-asset catalog (unchanged)


# ------------------------------------------------------------------------------
# Core CSV loader (single-asset flat OHLCV)
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

    # Convert timezone
    if isinstance(dt, pd.DatetimeIndex):
        if scheme.tz and (str(dt.tz) != scheme.tz):
            dt = dt.tz_convert(scheme.tz)
    else:
        dt = pd.DatetimeIndex(dt).tz_convert(scheme.tz)

    # Numeric coercion → float64
    num_cols = ["open", "high", "low", "close", "volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    out = df[num_cols].copy()
    out.index = pd.DatetimeIndex(dt, name="timestamp")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Drop rows with NaNs created during coercion
    out = out.dropna(how="any")

    # Sanity checks (after dropna)
    if len(out) == 0:
        raise ValueError(f"{path} contains no valid OHLCV rows.")
    if (out[["open", "high", "low", "close"]] < 0).any().any():
        raise ValueError("Negative price found — corrupted CSV?")
    if (out["high"] < out["low"]).any():
        raise ValueError("Row with high < low — corrupted OHLC?")
    if (out["volume"] < 0).any():
        raise ValueError("Negative volume found — corrupted CSV?")

    return out.astype("float64")


def resample_ohlcv(
    df: pd.DataFrame,
    rule: str,
    how: Literal["ohlc", "close"] = "ohlc",
    label: Literal["left", "right"] = "right",
    closed: Literal["left", "right"] = "right",
) -> pd.DataFrame:
    """
    Resample an OHLCV DataFrame to a new timeframe (e.g., '1h' -> '4h', '1D').
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
    if len(out) == 0:
        raise ValueError("Resampling produced an empty DataFrame.")
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
    """
    if dt_like is None:
        return None

    if isinstance(dt_like, datetime):
        if dt_like.tzinfo is None:
            dt_like = dt_like.replace(tzinfo=timezone.utc)
        return int(dt_like.timestamp() * 1000)

    if isinstance(dt_like, (np.integer, np.floating)):
        dt_like = dt_like.item()

    if isinstance(dt_like, (int, float)):
        x = int(round(dt_like))
        unit = _infer_epoch_unit(pd.Series([x]))
        ts = pd.to_datetime(x, unit=unit, utc=True)
        return int(ts.to_datetime64().astype("datetime64[ms]").astype("int64"))

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
    """
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end is not None else _to_ms(datetime.now(timezone.utc))
    if start_ms is None or end_ms is None or end_ms <= start_ms:
        raise ValueError("Invalid time range for Binance fetch (check start/end).")

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

        if len(data) < params["limit"]:
            break

        last_close_ms = int(data[-1][6])
        if last_close_ms >= end_ms:
            break
        cur = last_close_ms + 1
        time.sleep(request_pause_sec)

    if not rows:
        raise RuntimeError("No data returned for the requested range from Binance.")

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

    out = out[~out.index.duplicated(keep="last")].dropna(how="any")
    if len(out) == 0:
        raise RuntimeError("Binance returned rows, but all were invalid/NaN.")

    # Sanity checks
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

    Raises if no valid rows are returned.
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

    _write_ohlcv_csv(df, out_path)
    return out_path


def _write_ohlcv_csv(df: pd.DataFrame, path: Path) -> None:
    """Write OHLCV DataFrame to CSV with ISO-UTC timestamp. Refuses empty."""
    if len(df) == 0:
        raise ValueError("Refusing to write empty OHLCV CSV.")
    df_out = df.reset_index()
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df_out[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(
        path, index=False
    )


# ------------------------------------------------------------------------------
# Single-asset dataset catalog (numeric code)
# ------------------------------------------------------------------------------


def load_dataset_catalog(data_dir: str | Path = "data") -> pd.DataFrame:
    """
    Load (or initialize empty) single-asset dataset catalog.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / DATASET_CATALOG_NAME
    if not path.exists():
        cols = [
            "code",
            "symbol",
            "interval",
            "start",
            "end",
            "rows",
            "created_utc",
            "dataset_path",
        ]
        return pd.DataFrame(columns=cols)
    cat = pd.read_csv(path)
    # basic normalization
    if "code" in cat.columns:
        cat["code"] = pd.to_numeric(cat["code"], errors="coerce").astype("Int64")
    return cat


def _save_dataset_catalog(cat: pd.DataFrame, data_dir: Path) -> None:
    path = data_dir / DATASET_CATALOG_NAME
    cat.to_csv(path, index=False)


def _repr_time_for_catalog(x: str | int | float | datetime | None) -> str:
    """Human-friendly ISO-ish representation for the catalog."""
    if x is None:
        return ""
    if isinstance(x, datetime):
        if x.tzinfo is None:
            x = x.replace(tzinfo=timezone.utc)
        return x.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        ms = _to_ms(x)  # type: ignore[arg-type]
        ts = pd.to_datetime(ms, unit="ms", utc=True)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(x)


def _find_matching_code(
    cat: pd.DataFrame,
    symbol: str,
    interval: str,
    start: str | int | float | datetime | None,
    end: str | int | float | datetime | None,
) -> Optional[int]:
    if cat.empty:
        return None
    s = symbol.upper()
    i = interval
    st = _repr_time_for_catalog(start)
    en = _repr_time_for_catalog(end)
    m = cat[
        (cat["symbol"].str.upper() == s)
        & (cat["interval"] == i)
        & (cat["start"].fillna("") == st)
        & (cat["end"].fillna("") == en)
    ]
    if m.empty:
        return None
    return int(m.iloc[0]["code"])


def ensure_ohlcv_dataset(
    symbol: str,
    interval: str,
    start: str | int | float | datetime | None,
    end: str | int | float | datetime | None,
    *,
    data_dir: str | Path = "data",
    overwrite: bool = False,
    limit_per_call: int = 1000,
    request_pause_sec: float = 0.25,
) -> tuple[int, Path]:
    """
    Ensure an OHLCV dataset exists for the given (symbol, interval, start, end).
    - If the same spec exists in the catalog and file is present -> reuse.
    - Else fetch from Binance, assign a new numeric code, save data/<code>.csv,
      and append to catalog.

    Returns
    -------
    (code, path)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    cat = load_dataset_catalog(data_dir)
    existing_code = _find_matching_code(cat, symbol, interval, start, end)

    if existing_code is not None and not overwrite:
        path = data_dir / f"{existing_code}.csv"
        if path.exists():
            return existing_code, path
        # file missing: we will re-fetch and re-write using the same code
        code = existing_code
    else:
        # assign new code
        max_code = (
            int(cat["code"].max())
            if ("code" in cat.columns and not cat["code"].isna().all())
            else 0
        )
        code = max_code + 1

    # fetch
    df = fetch_binance_ohlcv_df(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        limit_per_call=limit_per_call,
        request_pause_sec=request_pause_sec,
    )

    # write dataset
    path = data_dir / f"{code}.csv"
    _write_ohlcv_csv(df, path)

    # upsert catalog row
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = {
        "code": int(code),
        "symbol": symbol.upper(),
        "interval": interval,
        "start": _repr_time_for_catalog(start),
        "end": _repr_time_for_catalog(end),
        "rows": int(len(df)),
        "created_utc": created,
        "dataset_path": str(path),
    }
    if existing_code is not None:
        cat.loc[cat["code"] == existing_code, :] = row
    else:
        cat = pd.concat([cat, pd.DataFrame([row])], ignore_index=True)

    _save_dataset_catalog(cat, data_dir)
    return code, path


def load_ohlcv_by_code(
    code: int | str,
    *,
    data_dir: str | Path = "data",
    tz: str = "UTC",
) -> pd.DataFrame:
    """
    Load a single-asset OHLCV dataset by numeric code from data/<code>.csv.
    """
    data_dir = Path(data_dir)
    path = data_dir / f"{int(code)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return load_ohlcv_csv(DataScheme(path=path, tz=tz))


# ------------------------------------------------------------------------------
# Portfolio dataset builder + catalog logging (unchanged style)
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
    Saves to <out_dir>/portfolio_<code>.csv and logs into portfolio_catalog.csv.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_clean = re.sub(r"[^A-Za-z0-9_\-\.]", "_", code).strip("_")
    out_path = out_dir / f"portfolio_{code_clean}.csv"
    catalog_path = out_dir / PORTFOLIO_CATALOG_NAME

    # Validate fields spec
    for sym, fields in symbols_fields.items():
        bad = [f for f in fields if f not in ALLOWED_FIELDS]
        if bad:
            raise ValueError(f"{sym}: unsupported fields requested: {bad}")

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
        sub = df.loc[:, list(fields)].copy()
        sub.columns = pd.MultiIndex.from_product(
            [[sym.upper()], list(fields)], names=["symbol", "field"]
        )
        frames.append(sub)

    if not frames:
        raise ValueError("symbols_fields is empty.")
    df_all = frames[0]
    for f in frames[1:]:
        df_all = df_all.join(f, how=join_how)

    df_all = df_all.sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="last")]
    if not isinstance(df_all.index, pd.DatetimeIndex) or df_all.index.tz is None:
        df_all.index = pd.DatetimeIndex(df_all.index, tz="UTC", name="timestamp")

    # Save
    df_out = df_all.reset_index()
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df_out.to_csv(out_path, index=False)

    # Append to portfolio catalog
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
    Append metadata to the portfolio catalog CSV (MultiIndex datasets).
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


def load_portfolio_csv(
    path: str | Path,
    *,
    tz: str = "UTC",
    fields_required: Sequence[str] | None = ("close",),
    flat_sep: str = "_",
) -> pd.DataFrame:
    """
    Load a multi-asset 'portfolio_<code>.csv' back into a MultiIndex-column DataFrame.
    Supports both MultiIndex headers and flat 'SYMBOL_field' headers.
    """
    path = Path(path)

    # Try MultiIndex header
    try:
        df = pd.read_csv(path, header=[0, 1])
        cols = df.columns
        if isinstance(cols, pd.MultiIndex):
            ts_candidates = [
                c
                for c in cols
                if (isinstance(c, tuple) and str(c[0]).lower() == "timestamp")
            ]
            if ts_candidates:
                ts_col = ts_candidates[0]
                ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                if ts.isna().any():
                    raise ValueError("Failed to parse timestamps (multiheader).")
                X = df.drop(columns=[ts_col])
                X.columns = pd.MultiIndex.from_tuples(
                    [(str(a).upper(), str(b)) for (a, b) in X.columns],
                    names=["symbol", "field"],
                )
                X.index = pd.DatetimeIndex(ts, name="timestamp")
                X = _normalize_multi_ohlcv(X, tz=tz, fields_required=fields_required)
                return X
    except Exception:
        pass

    # Fallback: flat header like 'BTCUSDT_close'
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Portfolio CSV must include a 'timestamp' column.")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Failed to parse timestamps in portfolio CSV.")

    cols = [c for c in df.columns if c != "timestamp"]
    tuples: list[tuple[str, str]] = []
    cols_ok: list[str] = []
    for c in cols:
        parts = str(c).rsplit(flat_sep, 1)
        if len(parts) == 2 and parts[1] in ALLOWED_FIELDS:
            tuples.append((parts[0].upper(), parts[1]))
            cols_ok.append(c)
    if not tuples:
        raise ValueError("Could not detect MultiIndex columns in portfolio CSV.")

    X = df[cols_ok].copy()
    X.columns = pd.MultiIndex.from_tuples(tuples, names=["symbol", "field"])
    X.index = pd.DatetimeIndex(ts, name="timestamp")
    X = _normalize_multi_ohlcv(X, tz=tz, fields_required=fields_required)
    return X


def _normalize_multi_ohlcv(
    X: pd.DataFrame, *, tz: str = "UTC", fields_required: Sequence[str] | None
) -> pd.DataFrame:
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


# Public exports
__all__ = [
    # CSV + loaders
    "DataScheme",
    "load_ohlcv_csv",
    "resample_ohlcv",
    # Binance
    "fetch_binance_ohlcv_df",
    "download_binance_ohlcv_csv",
    # Single-asset catalog
    "ensure_ohlcv_dataset",
    "load_ohlcv_by_code",
    "load_dataset_catalog",
    # Multi-asset builder
    "build_and_save_portfolio_dataset",
    "load_portfolio_csv",
]

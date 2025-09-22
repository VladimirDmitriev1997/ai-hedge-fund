from typing import Literal, Tuple

import numpy as np
import pandas as pd


def ensure_series(x: pd.Series, name: str | None = None) -> pd.Series:
    """
    Ensure the input is a float Series with a proper name.

    Parameters
    ----------
    x : pd.Series
        Input series-like (assumed already aligned as needed).
    name : str or None
        Optional replacement name.

    Returns
    -------
    pd.Series
        Float64 series aligned to the original index.
    """
    if not isinstance(x, pd.Series):
        raise TypeError("Expected a pandas Series.")
    out = pd.to_numeric(x, errors="coerce")
    if name is not None:
        out = out.rename(name)
    return out.astype(float)


def elapsed_from_index(
    idx: pd.DatetimeIndex,
    unit: str = "years",
    *,
    year_days: float = 365.25,
) -> float:
    """
    Compute elapsed time between the first and last timestamp of a DatetimeIndex
    in an arbitrary unit.

    Parameters
    ----------
    idx : pd.DatetimeIndex
        Time index (tz-aware or naive). Must contain at least two timestamps.
    unit : str, default "years"
        Target unit. One of:
        {"seconds","minutes","hours","days","weeks","months","years","bars"}.
        - "bars" returns the number of intervals: max(len(idx)-1, 0).
    year_days : float, default 365.25
        Days per year used to convert to "years" (and "months" via year_days/12).

    Returns
    -------
    float
        Elapsed time in the requested unit.

    Notes
    -----
    - For "months" we use an astronomical average: months_per_year = 12,
      so 1 month = (year_days / 12) days.
    - If idx has < 2 timestamps, returns 0.0.
    """
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("index must be a pandas DatetimeIndex")
    if len(idx) < 2:
        return 0.0

    if unit == "bars":
        return float(max(len(idx) - 1, 0))

    # total elapsed seconds between first and last timestamp
    delta_seconds = (idx[-1] - idx[0]).total_seconds()

    seconds_per_unit: dict[str, float] = {
        "seconds": 1.0,
        "minutes": 60.0,
        "hours": 3600.0,
        "days": 86400.0,
        "weeks": 7 * 86400.0,
        "months": (year_days / 12.0) * 86400.0,  # average month length
        "years": year_days * 86400.0,
    }

    unit = unit.lower()
    if unit not in seconds_per_unit:
        raise ValueError(
            f"unit must be one of {list(seconds_per_unit.keys()) + ['bars']}, got {unit!r}"
        )

    return float(delta_seconds / seconds_per_unit[unit])

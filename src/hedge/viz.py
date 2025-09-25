"""
Visualization utilities for strategies, holdings, and performance.

Design
------
- Pure matplotlib (no seaborn). No global state pollution.
- Functions return a matplotlib Figure and optionally save to disk.
- Handles both single-asset and multi-asset layouts.

Contents
--------
- plot_price_or_candles(...)
- plot_holdings(...)
- plot_equity_and_underwater(...)
- annotate_metrics_box(...)
- plot_features_grid(...)
- plot_single_asset_dashboard(...)
- plot_multi_asset_dashboard(...)

Conventions
-----------
- Index should be tz-aware DatetimeIndex (UTC recommended).
- Weights include a CASH column; risky columns are everything else.
- Equity is a Series aligned to the same index as weights (preferred).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

try:
    from hedge.portfolio import CASH_COL  # central constant if available
except Exception:
    CASH_COL = "CASH"


__all__ = [
    "plot_price_or_candles",
    "plot_holdings",
    "plot_equity_and_underwater",
    "annotate_metrics_box",
    "plot_features_grid",
    "plot_single_asset_dashboard",
    "plot_multi_asset_dashboard",
]


# ---------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------


def _format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )
    ax.grid(True, which="major", axis="both", alpha=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _percent_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))


def _ensure_series(x: Union[pd.Series, np.ndarray], name: str = "value") -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    arr = np.asarray(x)
    idx = pd.RangeIndex(len(arr))
    return pd.Series(arr, index=idx, name=name)


def _get_price_panel_for_asset(
    df: pd.DataFrame,
    asset: Optional[str] = None,
    field: str = "close",
) -> pd.DataFrame:
    """
    Extract OHLCV panel for one asset from:
    - Single-asset flat columns: open/high/low/close/volume
    - Multi-asset MultiIndex: (symbol, field)
    Returns DataFrame with subset of available columns among ['open','high','low','close','volume'].
    """
    cols = ("open", "high", "low", "close", "volume")

    # Multi-asset: columns MultiIndex (symbol, field)
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
        if asset is None:
            raise ValueError(
                "Provide 'asset' when plotting from a multi-asset DataFrame."
            )
        asset = str(asset).upper()
        keep = [c for c in cols if (asset, c) in df.columns]
        if not keep:
            raise KeyError(f"{asset}: no requested fields {cols} found in df.")
        panel = df.loc[:, pd.MultiIndex.from_product([[asset], keep])]
        panel.columns = [c[1] for c in panel.columns]  # flatten to single level fields
        return panel

    # Single-asset
    keep = [c for c in cols if c in df.columns]
    if not keep:
        raise KeyError(f"No OHLCV columns {cols} found.")
    return df[keep]


def _compute_underwater(equity: pd.Series) -> pd.Series:
    e = equity.astype(float)
    runup = e.cummax()
    dd = e / runup - 1.0
    return dd.rename("underwater")


# ---------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------


def plot_price_or_candles(
    df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    asset: Optional[str] = None,
    field: str = "close",
    mode: str = "candles",
    max_bars_for_candles: int = 3000,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot price (line) or candles for a given asset.

    Parameters
    ----------
    df : DataFrame
        OHLCV data. Single-asset flat columns OR MultiIndex (symbol, field).
    ax : plt.Axes, optional
        Axes to draw on. If None, creates a new Figure/Axes.
    asset : str, optional
        Required for MultiIndex input. Ignored for single-asset.
    field : {'close','open',...}, default 'close'
        Line plot field when mode='line' or when OHLC missing.
    mode : {'candles','line'}, default 'candles'
        Candles if OHLC present and N <= max_bars_for_candles, else line.
    max_bars_for_candles : int
        Safety cutoff to avoid super-slow rectangle plotting.
    title : str, optional
        Title for the chart.

    Returns
    -------
    matplotlib Figure
    """
    panel = _get_price_panel_for_asset(df, asset=asset, field=field)
    use_ax = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    idx = panel.index
    has_ohlc = all(c in panel.columns for c in ("open", "high", "low", "close"))
    plot_candles = (
        (mode == "candles") and has_ohlc and (len(panel) <= max_bars_for_candles)
    )

    if plot_candles:
        o, h, l, c = (panel["open"], panel["high"], panel["low"], panel["close"])
        x = mdates.date2num(pd.DatetimeIndex(idx).to_pydatetime())
        # Wick
        for xi, hi, lo in zip(x, h.values, l.values):
            ax.vlines(xi, lo, hi, linewidth=1.0, alpha=0.8)
        # Body
        width = (x[1] - x[0]) * 0.6 if len(x) > 1 else 0.6
        up = c >= o
        colors = np.where(up, "#2ca02c", "#d62728")  # green/red
        heights = (c - o).values
        bottoms = np.minimum(o.values, c.values)
        for xi, btm, hgt, col in zip(x, bottoms, np.abs(heights), colors):
            rect = Rectangle(
                (xi - width / 2, btm),
                width,
                hgt if hgt != 0 else 1e-9,
                facecolor=col,
                edgecolor=col,
                alpha=0.8,
            )
            ax.add_patch(rect)
        ax.set_xlim(x.min() - width, x.max() + width)
        ax.set_ylabel("Price")
    else:
        # Fallback / explicit line
        y = panel[field] if field in panel.columns else panel.iloc[:, 0]
        ax.plot(idx, y.values, linewidth=1.4)
        ax.set_ylabel(f"{field.capitalize()}")

    if title:
        ax.set_title(title, fontsize=12, pad=8)
    _format_time_axis(ax)
    if not use_ax:
        fig.tight_layout()
    return fig


def plot_holdings(
    holdings: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    kind: str = "lines",  # 'lines' | 'stacked'
    top_n: int = 8,
    title: Optional[str] = "Holdings (weights)",
) -> plt.Figure:
    """
    Plot portfolio holdings/weights over time.

    Parameters
    ----------
    holdings : DataFrame
        Columns include CASH; risky columns are others. Rows sum ~ 1.
    ax : plt.Axes, optional
        Axes to draw on. If None, creates a new Figure/Axes.
    kind : {'lines','stacked'}, default 'lines'
        Plot style. 'stacked' is better for many assets (stacked area of riskies).
    top_n : int
        For stacked: show top_n assets by average |weight|; rest grouped as 'Other'.
    title : str, optional
        Title.

    Returns
    -------
    matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.figure

    cols = [c for c in holdings.columns if str(c) != CASH_COL]
    idx = holdings.index

    if kind == "stacked" and len(cols) > 1:
        risky = holdings[cols].copy()
        # Rank by average absolute exposure
        avg_abs = risky.abs().mean().sort_values(ascending=False)
        keep = list(avg_abs.head(max(1, top_n)).index)
        rest = [c for c in cols if c not in keep]
        stack = pd.DataFrame(index=idx)
        for c in keep:
            stack[c] = risky[c]
        if rest:
            stack["Other"] = risky[rest].sum(axis=1)
        ax.stackplot(
            idx,
            [stack[c].values for c in stack.columns],
            labels=stack.columns,
            alpha=0.8,
        )
        ax.legend(loc="upper left", ncols=2, fontsize=8, frameon=False)
    else:
        for c in cols + ([CASH_COL] if CASH_COL in holdings.columns else []):
            ax.plot(idx, holdings[c].values, linewidth=1.2, label=str(c))
        if len(cols) <= 6:
            ax.legend(loc="upper left", ncols=2, fontsize=8, frameon=False)

    _percent_axis(ax)
    ax.set_ylabel("Weight")
    if title:
        ax.set_title(title, fontsize=11, pad=6)
    _format_time_axis(ax)
    fig.tight_layout()
    return fig


def plot_equity_and_underwater(
    equity: Union[pd.Series, np.ndarray],
    *,
    ax_equity: Optional[plt.Axes] = None,
    ax_under: Optional[plt.Axes] = None,
    sharex: bool = True,
    title_equity: str = "Equity",
    title_under: str = "Underwater",
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Plot equity line and underwater (drawdown) area on two aligned axes.

    Returns
    -------
    (fig, ax_equity, ax_under)
    """
    e = _ensure_series(equity, name="equity").astype(float)

    new_fig = ax_equity is None or ax_under is None
    if new_fig:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(12, 5),
            sharex=True if sharex else False,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig = ax_equity.figure
        ax1, ax2 = ax_equity, ax_under

    ax1.plot(e.index, e.values, linewidth=1.6)
    ax1.set_title(title_equity, fontsize=12, pad=6)
    ax1.set_ylabel("Equity")
    _format_time_axis(ax1)

    uw = _compute_underwater(e)
    ax2.fill_between(uw.index, uw.values, 0.0, step="pre", alpha=0.6)
    ax2.set_title(title_under, fontsize=11, pad=4)
    ax2.set_ylabel("Drawdown")
    _format_time_axis(ax2)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.tight_layout()
    return fig, ax1, ax2


def annotate_metrics_box(
    ax: plt.Axes,
    metrics: Mapping[str, Union[float, int, str]],
    *,
    loc: str = "upper left",
    fontsize: int = 9,
    title: Optional[str] = "Metrics",
) -> None:
    """
    Annotate axis with a metrics text box.

    Parameters
    ----------
    ax : plt.Axes
        Target axis.
    metrics : mapping
        Key->value pairs (ROI/CAGR/Sharpe/MDD/Calmar/Turnover...).
    loc : {'upper left','upper right','lower left','lower right'}
        Box location.
    fontsize : int
        Font size.
    title : str, optional
        Box title.
    """
    if not metrics:
        return

    kv = []
    for k, v in metrics.items():
        if isinstance(v, (int, np.integer)):
            kv.append(f"{k}: {int(v)}")
        elif isinstance(v, (float, np.floating)):
            # format percentages nicely if obvious
            if any(s in k.lower() for s in ("roi", "cagr", "mdd", "dd", "calmar")):
                kv.append(f"{k}: {v:,.2%}")
            elif "sharpe" in k.lower():
                kv.append(f"{k}: {v:,.2f}")
            else:
                kv.append(f"{k}: {v:,.4f}")
        else:
            kv.append(f"{k}: {v}")
    text = (title + "\n" if title else "") + "\n".join(kv)

    bbox = dict(facecolor="#f5f5f7", edgecolor="#dddddd", boxstyle="round,pad=0.5")
    anchors = {
        "upper left": dict(x=0.02, y=0.98, ha="left", va="top"),
        "upper right": dict(x=0.98, y=0.98, ha="right", va="top"),
        "lower left": dict(x=0.02, y=0.02, ha="left", va="bottom"),
        "lower right": dict(x=0.98, y=0.02, ha="right", va="bottom"),
    }
    kw = anchors.get(loc, anchors["upper left"])
    ax.text(transform=ax.transAxes, s=text, fontsize=fontsize, **kw, bbox=bbox)


def plot_features_grid(
    features: Mapping[str, Union[pd.Series, np.ndarray]],
    *,
    ncols: int = 3,
    sharex: bool = True,
    height_per_row: float = 1.2,
    title: Optional[str] = "Features",
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """
    Plot small multiple panels for selected features.

    Parameters
    ----------
    features : mapping
        name -> series/array aligned in time.
    ncols : int
        Number of columns in the grid.
    sharex : bool
        Share X (time) across panels.
    height_per_row : float
        Figure height per row.

    Returns
    -------
    (fig, axes)
    """
    if not features:
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.axis("off")
        return fig, [ax]

    items = list(features.items())
    n = len(items)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, max(1.0, nrows * height_per_row)), sharex=sharex
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])

    axes = axes.reshape(nrows, ncols)
    k = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if k < n:
                name, series = items[k]
                s = _ensure_series(series, name=name).astype(float)
                ax.plot(s.index, s.values, linewidth=1.0)
                ax.set_title(str(name), fontsize=9, pad=3)
                _format_time_axis(ax)
            else:
                ax.axis("off")
            k += 1

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        fig.tight_layout()
    return fig, axes.ravel().tolist()


# ---------------------------------------------------------------------
# Dashboards
# ---------------------------------------------------------------------


def plot_single_asset_dashboard(
    df: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    equity: Optional[pd.Series] = None,
    features: Optional[Mapping[str, Union[pd.Series, np.ndarray]]] = None,
    metrics: Optional[Mapping[str, Union[float, int, str]]] = None,
    asset: Optional[str] = None,
    price_mode: str = "candles",  # 'candles' | 'line'
    figsize: Tuple[float, float] = (13, 9),
    savepath: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Single-asset layout:
      [Price/Candles]
      [Equity   (optional)]
      [Holdings]
      [Feature mini-panels (optional)]
    """
    # Build figure with Gridspec
    n_feature_rows = 1 if features else 0
    height_ratios = [3, 2 if equity is not None else 0, 1.8, n_feature_rows]
    height_ratios = [h for h in height_ratios if h > 0]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        len(height_ratios), 1, height_ratios=height_ratios, hspace=0.35
    )

    row = 0
    # Price/Candles
    ax_price = fig.add_subplot(gs[row, 0])
    row += 1
    plot_price_or_candles(
        df, ax=ax_price, asset=asset, mode=price_mode, title=f"Price — {asset or ''}"
    )

    # Equity + Underwater
    if equity is not None:
        ax_eq = fig.add_subplot(gs[row, 0])
        row += 1
        ax_under = fig.add_subplot(gs[row, 0], sharex=ax_eq)
        row += 0  # will be replaced by function
        # Remove the placeholder axis and let helper create both
        ax_under.remove()
        fig, ax_eq, ax_under = plot_equity_and_underwater(
            equity, ax_equity=ax_eq, ax_under=None, sharex=True
        )

        if metrics:
            annotate_metrics_box(ax_eq, metrics, loc="upper left")

    # Holdings
    ax_hold = fig.add_subplot(gs[row, 0])
    row += 1
    plot_holdings(weights, ax=ax_hold, kind="lines")

    # Features grid
    if features:
        # place as its own figure and then embed? Simpler: create a small grid below in the same fig.
        # We will create a new axes area in the same gridspec row.
        feature_fig, _ = plot_features_grid(
            features, ncols=3, sharex=True, height_per_row=1.0, title=None
        )
        # Insert artists: easier approach is to draw side-by-side; here we just close the temp fig and inform user
        plt.close(feature_fig)
        # Instead, reuse the holdings axis to get x-range and create small inset axes
        # For simplicity and robustness across Matplotlib versions, create a new axis row and plot all features overlapped with legend.
        ax_feat = fig.add_subplot(gs[row, 0])
        for name, ser in features.items():
            s = _ensure_series(ser, name=str(name)).astype(float)
            ax_feat.plot(s.index, s.values, linewidth=0.9, label=str(name))
        ax_feat.set_title("Features", fontsize=11, pad=6)
        _format_time_axis(ax_feat)
        ax_feat.legend(loc="upper left", ncols=3, fontsize=8, frameon=False)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=140)
    if show:
        plt.show()
    return fig


def plot_multi_asset_dashboard(
    weights: pd.DataFrame,
    *,
    equity: Optional[pd.Series] = None,
    metrics: Optional[Mapping[str, Union[float, int, str]]] = None,
    holdings_kind: str = "stacked",
    top_n: int = 10,
    figsize: Tuple[float, float] = (13, 7),
    savepath: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Multi-asset layout (no single price pane):
      [Equity + Underwater (optional)]
      [Holdings (stacked area by default)]
      [Metrics box (overlay on equity if provided)]
    """
    # Determine rows
    nrows = 1 + (1 if equity is not None else 0)
    height_ratios = [2, 2] if equity is not None else [2]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        len(height_ratios), 1, height_ratios=height_ratios, hspace=0.35
    )

    row = 0
    if equity is not None:
        ax_eq = fig.add_subplot(gs[row, 0])
        row += 1
        ax_under = fig.add_subplot(gs[row, 0], sharex=ax_eq)
        row += 0
        ax_under.remove()
        fig, ax_eq, ax_under = plot_equity_and_underwater(
            equity, ax_equity=ax_eq, ax_under=None, sharex=True
        )
        if metrics:
            annotate_metrics_box(ax_eq, metrics, loc="upper left")

    ax_hold = fig.add_subplot(gs[row, 0])
    row += 1
    plot_holdings(weights, ax=ax_hold, kind=holdings_kind, top_n=top_n)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=140)
    if show:
        plt.show()
    return fig


def plot_run_simple(
    df: pd.DataFrame,
    weights: pd.DataFrame,
    equity: pd.Series,
    *,
    asset: str | None = None,
    price_field: str = "close",
    top_n_weights: int = 8,
    figsize: tuple[float, float] = (13, 10),
    title: str | None = None,
    metrics: Mapping[str, float | int | str] | None = None,
    savepath: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Minimal stacked visualization for a single run (train or test).

    Panels (one per row, no overlays):
      1) Price (close)
      2) Equity
      3) Underwater (drawdown)
      4) Weights (lines)
      5) Metrics (text panel, optional)

    df:
        OHLCV or portfolio frame; if MultiIndex (symbol, field), provide `asset`.
    weights:
        DataFrame with columns including CASH; indexed like df/equity.
    equity:
        Series aligned to time index.
    asset:
        Required when df has MultiIndex columns (choose which asset's price to show).
    """
    # ----- extract price series -----
    if isinstance(df.columns, pd.MultiIndex):
        if not asset:
            raise ValueError("Provide `asset` when df has MultiIndex columns.")
        asset = str(asset).upper()
        if (asset, price_field) not in df.columns:
            raise KeyError(f"Missing {asset}.{price_field} in df.")
        price = df[(asset, price_field)].astype(float)
        price_name = asset
    else:
        if price_field not in df.columns:
            # fallback to first numeric column
            num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if not num_cols:
                raise KeyError("No numeric price column found.")
            price_field = num_cols[0]
        price = df[price_field].astype(float)
        price_name = price_field

    price = price.reindex(equity.index)

    # ----- underwater -----
    runup = equity.astype(float).cummax()
    underwater = (equity.astype(float) / runup) - 1.0

    # ----- set up figure -----
    nrows = 5 if metrics else 4
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)

    # 1) Price
    ax = axes[0]
    ax.plot(price.index, price.values, linewidth=1.4)
    ax.set_ylabel("Close")
    ax.set_title(title or f"{price_name}", fontsize=12)
    _format_time_axis(ax)

    # 2) Equity
    ax = axes[1]
    ax.plot(equity.index, equity.values, linewidth=1.6)
    ax.set_ylabel("Equity")
    _format_time_axis(ax)

    # 3) Underwater
    ax = axes[2]
    ax.fill_between(underwater.index, underwater.values, 0.0, step="pre", alpha=0.65)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    _format_time_axis(ax)

    # 4) Weights (lines; top-N riskies + CASH if present)
    ax = axes[3]
    cols = [c for c in weights.columns if str(c) != CASH_COL]
    # rank by avg |w|
    avg_abs = (
        weights[cols].abs().mean().sort_values(ascending=False)
        if cols
        else pd.Series(dtype=float)
    )
    keep = list(avg_abs.head(max(1, top_n_weights)).index) if len(avg_abs) else []
    plot_cols = keep + ([CASH_COL] if CASH_COL in weights.columns else [])
    for c in plot_cols:
        ax.plot(weights.index, weights[c].values, linewidth=1.1, label=str(c))
    if plot_cols:
        ax.legend(loc="upper left", ncols=3, fontsize=8, frameon=False)
    ax.set_ylabel("Weight")
    _percent_axis(ax)
    _format_time_axis(ax)

    # 5) Metrics (text only)
    if metrics:
        ax = axes[4]
        ax.axis("off")
        lines = []
        for k, v in metrics.items():
            if isinstance(v, (float, np.floating)):
                if any(
                    s in k.lower() for s in ("roi", "cagr", "drawdown", "mdd", "calmar")
                ):
                    lines.append(f"{k}: {v:,.2%}")
                elif "sharpe" in k.lower():
                    lines.append(f"{k}: {v:,.2f}")
                else:
                    lines.append(f"{k}: {v:,.4f}")
            else:
                lines.append(f"{k}: {v}")
        text = "\n".join(lines)
        ax.text(
            0.01,
            0.98,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
        )

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    return fig

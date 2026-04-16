"""
Reusable plotting and diagnostic helpers for EDA of the London Smart Meters dataset.

All functions return a matplotlib Figure so callers can save or display them.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time-series helpers
# ---------------------------------------------------------------------------

def plot_time_series(
    df: pd.DataFrame,
    household_id: str,
    value_col: str = "energy(kWh/hh)",
    timestamp_col: str = "tstp",
    title: Optional[str] = None,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """Plot energy consumption over time for a single household."""
    subset = df[df["LCLid"] == household_id].copy()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(subset[timestamp_col], subset[value_col], linewidth=0.6, alpha=0.85)
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    ax.set_title(title or f"Energy consumption — {household_id}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_multiple_households(
    df: pd.DataFrame,
    household_ids: Sequence[str],
    value_col: str = "energy(kWh/hh)",
    timestamp_col: str = "tstp",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Overlay energy time series for several households on one axes."""
    fig, ax = plt.subplots(figsize=figsize)
    for hid in household_ids:
        subset = df[df["LCLid"] == hid]
        ax.plot(subset[timestamp_col], subset[value_col], linewidth=0.5, alpha=0.7, label=hid)
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    ax.set_title("Energy consumption — selected households")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=7, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_acf_pacf(
    df: pd.DataFrame,
    household_ids: Sequence[str],
    lags: int = 100,
    timestamp_col: str = "tstp",
    value_col: str = "energy(kWh/hh)",
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """Plot ACF and PACF side-by-side for each household (n rows × 2 cols).

    Args:
        df:             Halfhourly DataFrame sorted by (LCLid, tstp).
        household_ids:  Household LCLids to plot (up to 3 recommended).
        lags:           Number of lags.  Default 100 shows past the daily peak (lag 48).
        timestamp_col:  Timestamp column name.
        value_col:      Energy column name.
        figsize:        (width, height-per-row) tuple.

    Returns:
        matplotlib Figure.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    ids = list(household_ids)
    fig, axes = plt.subplots(
        nrows=len(ids),
        ncols=2,
        figsize=(figsize[0], figsize[1] * len(ids)),
        constrained_layout=True,
    )
    if len(ids) == 1:
        axes = axes[np.newaxis, :]

    for row_idx, hid in enumerate(ids):
        series = (
            df[df["LCLid"] == hid]
            .sort_values(timestamp_col)[value_col]
            .dropna()
            .reset_index(drop=True)
        )
        plot_acf(series, lags=lags, ax=axes[row_idx, 0], title=f"ACF — {hid}")
        plot_pacf(series, lags=lags, ax=axes[row_idx, 1], title=f"PACF — {hid}", method="ywm")

    fig.suptitle(
        f"ACF / PACF  ({lags} lags)",
        fontsize=11,
        y=1.01,
    )
    return fig


# Zoom levels: (label, half-window timedelta, date formatter, major locator)
_ZOOM_LEVELS = [
    ("3 months", pd.Timedelta(days=45),  mdates.DateFormatter("%d %b"),  mdates.WeekdayLocator(byweekday=0)),
    ("1 week",   pd.Timedelta(days=3.5), mdates.DateFormatter("%d %b"),  mdates.DayLocator()),
    ("1 day",    pd.Timedelta(hours=12), mdates.DateFormatter("%H:%M"),   mdates.HourLocator(byhour=range(0, 24, 4))),
]


def plot_halfhourly_zoom(
    df: pd.DataFrame,
    household_ids: Sequence[str],
    anchor: Optional[pd.Timestamp] = None,
    value_col: str = "energy(kWh/hh)",
    timestamp_col: str = "tstp",
    figsize: tuple = (18, 14),
) -> plt.Figure:
    """
    Plot 5 households at three zoom levels (3 months, 1 week, 1 day).

    Layout: 5 rows (one per household) × 3 columns (one per zoom level).

    Args:
        df:             Halfhourly DataFrame sorted by (LCLid, tstp).
        household_ids:  Exactly 5 household LCLids to plot.
        anchor:         Centre timestamp for each window.  Defaults to the
                        dataset midpoint.
        value_col:      Energy column name.
        timestamp_col:  Timestamp column name.
        figsize:        Overall figure size.

    Returns:
        matplotlib Figure.
    """
    ids = list(household_ids)[:5]

    if anchor is None:
        t_min = df[timestamp_col].min()
        t_max = df[timestamp_col].max()
        anchor = t_min + (t_max - t_min) / 2

    fig, axes = plt.subplots(
        nrows=len(ids),
        ncols=len(_ZOOM_LEVELS),
        figsize=figsize,
        constrained_layout=True,
    )
    # Ensure 2-D indexing even for a single household
    if len(ids) == 1:
        axes = axes[np.newaxis, :]

    for col_idx, (zoom_label, half_win, date_fmt, major_loc) in enumerate(_ZOOM_LEVELS):
        t_start = anchor - half_win
        t_end   = anchor + half_win

        for row_idx, hid in enumerate(ids):
            ax = axes[row_idx, col_idx]

            subset = df[
                (df["LCLid"] == hid) &
                (df[timestamp_col] >= t_start) &
                (df[timestamp_col] <= t_end)
            ]

            if subset.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.plot(
                    subset[timestamp_col],
                    subset[value_col],
                    linewidth=0.8,
                    color=f"C{row_idx}",
                )
                ax.set_xlim(t_start, t_end)

            ax.xaxis.set_major_locator(major_loc)
            ax.xaxis.set_major_formatter(date_fmt)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
            ax.tick_params(axis="y", labelsize=7)

            # Row labels (left-most column only)
            if col_idx == 0:
                ax.set_ylabel(hid, fontsize=7, rotation=0, labelpad=60, va="center")

            # Column headers (top row only)
            if row_idx == 0:
                ax.set_title(zoom_label, fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Half-hourly energy — {len(ids)} households  (anchor: {anchor.date()})",
        fontsize=11,
        y=1.01,
    )
    return fig


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def plot_daily_profile(
    df: pd.DataFrame,
    timestamp_col: str = "tstp",
    value_col: str = "energy(kWh/hh)",
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Average intra-day energy profile (mean ± std per 30-min slot)."""
    tmp = df[[timestamp_col, value_col]].copy()
    tmp["slot"] = tmp[timestamp_col].dt.hour * 2 + tmp[timestamp_col].dt.minute // 30
    profile = tmp.groupby("slot")[value_col].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=figsize)
    x = profile.index
    ax.plot(x, profile["mean"], linewidth=1.5, label="mean")
    ax.fill_between(
        x,
        profile["mean"] - profile["std"],
        profile["mean"] + profile["std"],
        alpha=0.2,
        label="±1 std",
    )
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    ax.set_xlabel("Time of day")
    ax.set_ylabel(value_col)
    ax.set_title("Average daily energy profile (all households)")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Missing-data diagnostics
# ---------------------------------------------------------------------------

def plot_missing_heatmap(
    df: pd.DataFrame,
    max_cols: int = 50,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Heatmap of missing values per column.

    For wide datasets (e.g. hhblock) only the first *max_cols* columns are shown.
    """
    miss_pct = df.isnull().mean().sort_values(ascending=False)
    miss_pct = miss_pct.head(max_cols)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(miss_pct)), miss_pct.values * 100, color="tomato", alpha=0.8)
    ax.set_xticks(range(len(miss_pct)))
    ax.set_xticklabels(miss_pct.index, rotation=90, fontsize=7)
    ax.set_ylabel("Missing (%)")
    ax.set_title("Missing values per column")
    fig.tight_layout()
    return fig


def print_dataset_summary(df: pd.DataFrame, dataset_type: str, metadata: dict) -> None:
    """Print a concise diagnostic summary to stdout."""
    ts_col = "tstp" if dataset_type == "halfhourly_dataset" else "day"
    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_type}")
    print(f"Shape   : {df.shape}")
    print(f"Households : {df['LCLid'].nunique()}")
    if ts_col in df.columns:
        print(f"Time range : {df[ts_col].min()}  to  {df[ts_col].max()}")
    print(f"Blocks loaded  : {len(metadata.get('blocks_loaded', []))}")
    print(f"Blocks failed  : {len(metadata.get('blocks_failed', []))}")
    print(f"Duplicates removed : {metadata.get('duplicates_removed', 0)}")
    print(f"Invalid records    : {metadata.get('invalid_records', 0)}")
    print(f"Memory usage : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"{'='*60}\n")

"""
Feature builder functions for half-hourly energy time series.

Each builder has the signature:
    (batch: pd.DataFrame, value_col: str) -> pd.DataFrame

It receives a batch of households (MultiIndex: LCLid × tstp), computes new
columns using only past information within each household, and returns a
DataFrame of those columns aligned to the same index.

To add a new feature set, implement a function with the same signature and
append a (partial of that) function to FEATURE_BUILDERS in task03.
"""

from __future__ import annotations

import logging
from functools import partial

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_lag_features(
    batch: pd.DataFrame,
    value_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """Lag the energy series by each value in *lags* within each household."""
    cols = {
        f"lag_{lag}": batch.groupby(level="LCLid")[value_col].shift(lag)
        for lag in lags
    }
    new_cols = list(cols.keys())
    logger.debug("build_lag_features created %d columns: %s", len(new_cols), new_cols)
    return pd.DataFrame(cols, index=batch.index)


def build_rolling_features(
    batch: pd.DataFrame,
    value_col: str,
    windows: list[int],
) -> pd.DataFrame:
    """
    Rolling mean and std over each window within each household.

    closed="left" excludes the current observation so the statistics are
    strictly causal (computed from past-only data).  min_periods=1 allows
    partial windows at the start of each household's history rather than
    producing NaNs for the first *window* rows.
    """
    cols = {}
    for window in windows:
        rolled = (
            batch.groupby(level="LCLid")[value_col]
            .rolling(window, min_periods=1, closed="left")
        )
        # reset_index(level=0, drop=True) removes the redundant LCLid level
        # that groupby().rolling() prepends to the result index
        cols[f"rolling_{window}_mean"] = rolled.mean().reset_index(level=0, drop=True)
        cols[f"rolling_{window}_std"]  = rolled.std() .reset_index(level=0, drop=True)

    new_cols = list(cols.keys())
    logger.debug("build_rolling_features created %d columns: %s", len(new_cols), new_cols)
    return pd.DataFrame(cols, index=batch.index)


def build_ewma_features(
    batch: pd.DataFrame,
    value_col: str,
    spans: list[int],
) -> pd.DataFrame:
    """
    Exponentially weighted moving average (EWMA) at multiple spans.

    The series is shifted by 1 within each household before computing EWM so
    that each row uses only past observations (strictly causal).
    """
    cols = {}
    # Shift once, then compute EWM for each span
    shifted = batch.groupby(level="LCLid")[value_col].shift(1)
    for span in spans:
        ewma = (
            shifted
            .groupby(level="LCLid")
            .transform(lambda s: s.ewm(span=span, min_periods=1).mean())
        )
        cols[f"ewma_span_{span}"] = ewma

    new_cols = list(cols.keys())
    logger.debug("build_ewma_features created %d columns: %s", len(new_cols), new_cols)
    return pd.DataFrame(cols, index=batch.index)


def build_calendar_features(
    batch: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """
    Temporal/calendar features derived from the timestamp index.

    These are entirely causal (no future leakage) because they depend only on
    the current slot's timestamp, not on the target value.

    Features:
        hour_of_day  — 0–23
        day_of_week  — 0 (Mon) – 6 (Sun)
        month        — 1–12
        is_weekend   — 1 if Saturday or Sunday, else 0
    """
    tstp = batch.index.get_level_values("tstp")
    cols = {
        "hour_of_day": tstp.hour,
        "day_of_week": tstp.dayofweek,
        "month":       tstp.month,
        "is_weekend":  (tstp.dayofweek >= 5).astype(np.int8),
    }
    new_cols = list(cols.keys())
    logger.debug("build_calendar_features created %d columns: %s", len(new_cols), new_cols)
    return pd.DataFrame(cols, index=batch.index)


def build_fourier_features(
    batch: pd.DataFrame,
    value_col: str,
    periods: list[float] | None = None,
    n_terms: int = 2,
) -> pd.DataFrame:
    """
    Fourier (sine/cosine) features for each period in *periods*.

    The position within each period is computed purely from the timestamp so
    there is no target leakage.

    Default periods (in half-hour slots):
        48  — daily   (24 h × 2 slots/h)
        336 — weekly  (7 days × 48 slots/day)

    For each period P and term k (1 … n_terms):
        fourier_p{P}_sin_{k}  = sin(2π k t / P)
        fourier_p{P}_cos_{k}  = cos(2π k t / P)

    where *t* is the slot position within the period cycle derived from the
    timestamp (not a running index), so the features are shift-invariant and
    generalize across households.
    """
    if periods is None:
        periods = [48.0, 336.0]

    tstp = batch.index.get_level_values("tstp")
    # Half-hour slot within the week: 0 … 335
    weekly_slot = (tstp.dayofweek * 48 + tstp.hour * 2 + tstp.minute // 30).astype(float)

    cols = {}
    for period in periods:
        period = float(period)
        if period == 48.0:
            # Position within daily cycle
            t = (tstp.hour * 2 + tstp.minute // 30).astype(float)
        elif period == 336.0:
            # Position within weekly cycle
            t = weekly_slot
        else:
            # Generic: position within arbitrary-length cycle using weekly slot mod period
            t = weekly_slot % period

        for k in range(1, n_terms + 1):
            angle = 2.0 * np.pi * k * t / period
            cols[f"fourier_p{int(period)}_sin_{k}"] = np.sin(angle)
            cols[f"fourier_p{int(period)}_cos_{k}"] = np.cos(angle)

    new_cols = list(cols.keys())
    logger.debug("build_fourier_features created %d columns: %s", len(new_cols), new_cols)
    return pd.DataFrame(cols, index=batch.index)

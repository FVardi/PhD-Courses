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

from functools import partial

import pandas as pd


def build_lag_features(
    batch: pd.DataFrame,
    value_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """Lag the energy series by each value in *lags* within each household."""
    return pd.DataFrame(
        {
            f"lag_{lag}": batch.groupby(level="LCLid")[value_col].shift(lag)
            for lag in lags
        },
        index=batch.index,
    )


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

    return pd.DataFrame(cols, index=batch.index)

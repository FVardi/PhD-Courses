"""
Reindexing and resampling helpers for time-series household data.
"""

from __future__ import annotations

import pandas as pd


def build_resampled_grid(
    df: pd.DataFrame,
    id_col: str = "LCLid",
    ts_col: str = "tstp",
    value_col: str = "energy(kWh/hh)",
    freq: str = "30min",
) -> pd.DataFrame:
    """
    Resample a flat household DataFrame onto a regular time grid.

    Each household is independently resampled so that every expected slot
    exists; slots with no observations become NaN.  The result has a
    MultiIndex (id_col, ts_col).

    Args:
        df:         Flat DataFrame with id_col, ts_col, and value_col columns.
        id_col:     Household identifier column.
        ts_col:     Timestamp column (will be parsed to datetime if needed).
        value_col:  Energy / value column to resample.
        freq:       Target grid frequency (default "30min").

    Returns:
        DataFrame with MultiIndex (id_col, ts_col) and a single value_col column.
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col)
    return (
        df.groupby(id_col)
        .apply(lambda g: g[[value_col]].resample(freq).mean())
    )

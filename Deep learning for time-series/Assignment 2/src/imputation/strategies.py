"""
Imputation strategies for half-hourly energy time series.

  naive_impute           – per-household Series transform; fill from lag-48
  seasonal_mean_impute   – full-DataFrame function; causal expanding slot mean
"""

from __future__ import annotations

import pandas as pd


def naive_impute(series: pd.Series) -> pd.Series:
    """Fill NaNs with the value from the same half-hour slot 24 hours earlier (lag-48)."""
    return series.fillna(series.shift(48))


def seasonal_mean_impute(
    hh_df: pd.DataFrame,
    id_col: str = "LCLid",
    value_col: str = "energy(kWh/hh)",
) -> pd.Series:
    """
    Fill NaNs with the causal expanding mean of the same half-hour slot.

    For each missing slot at time t the fill value is the mean of all *previous*
    non-missing observations at that (household, slot) pair.  No future data is
    used.  Slots with no prior observations remain NaN.

    Fully vectorised: uses groupby cumsum and shift — no Python-level loops.
    """
    s     = hh_df[value_col]
    tstp  = s.index.get_level_values("tstp")
    slot  = tstp.hour * 2 + tstp.minute // 30
    lclid = s.index.get_level_values(id_col)
    groups = [lclid, slot]

    # Inclusive running sum / count of non-NaN values within each (household, slot) group,
    # then shifted by 1 to exclude the current observation → strictly causal mean.
    causal_sum   = s.groupby(groups).cumsum().groupby(groups).shift(1)
    notna_count  = s.notna().groupby(groups).cumsum().groupby(groups).shift(1)
    causal_mean  = causal_sum / notna_count

    return s.fillna(causal_mean)

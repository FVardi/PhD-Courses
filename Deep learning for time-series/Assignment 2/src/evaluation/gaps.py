"""
Gap detection helpers for time-series data.

Functions for raw data (time-delta approach):
  find_gaps            – flat table of every gap per household
  gaps_per_household   – per-household summary (count, total missing, longest gap)
  summarise_gaps       – dataset-level missingness rate + gap-length distribution

Functions for resampled grids (NaN-run approach):
  find_gap_lengths     – list of consecutive NaN run lengths for a single series
  gap_length_dataframe – flat DataFrame of gap lengths across all households
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Raw-data gap detection (time-delta approach)
# ---------------------------------------------------------------------------

def find_gaps(
    df: pd.DataFrame,
    id_col: str = "LCLid",
    ts_col: str = "tstp",
    freq: str = "30min",
) -> pd.DataFrame:
    """
    Return every gap in every household's time series as a flat DataFrame.

    A gap is any consecutive pair of observations whose time distance exceeds
    one expected period (*freq*).

    Columns returned:
        id_col          household identifier
        gap_start       last observed timestamp before the gap
        gap_end         first observed timestamp after the gap
        missing_periods number of expected periods absent
        gap_duration    gap_end - gap_start  (timedelta)
    """
    expected = pd.Timedelta(freq)
    records = []

    for hid, grp in df.groupby(id_col, sort=False):
        ts = grp[ts_col].sort_values().reset_index(drop=True)
        deltas = ts.diff()

        gap_idx = deltas.index[deltas > expected]

        for i in gap_idx:
            gap_start = ts.iloc[i - 1]
            gap_end   = ts.iloc[i]
            duration  = gap_end - gap_start
            missing   = int(round(duration / expected)) - 1
            records.append(
                {
                    id_col: hid,
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                    "missing_periods": missing,
                    "gap_duration": duration,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[id_col, "gap_start", "gap_end", "missing_periods", "gap_duration"]
        )

    return pd.DataFrame(records).sort_values([id_col, "gap_start"]).reset_index(drop=True)


def gaps_per_household(
    df: pd.DataFrame,
    id_col: str = "LCLid",
    ts_col: str = "tstp",
    freq: str = "30min",
) -> pd.DataFrame:
    """
    Summarise gap statistics per household.

    Columns returned:
        id_col              household identifier
        n_gaps              number of distinct gaps
        total_missing       total missing periods (sum of gap lengths)
        longest_gap         largest single missing_periods value
        total_observed      number of rows present in *df* for this household
        expected_periods    total periods expected between first and last observation
        missingness_rate    total_missing / expected_periods
    """
    expected = pd.Timedelta(freq)
    gap_df = find_gaps(df, id_col=id_col, ts_col=ts_col, freq=freq)

    if gap_df.empty:
        gap_agg = pd.DataFrame(columns=[id_col, "n_gaps", "total_missing", "longest_gap"])
    else:
        gap_agg = (
            gap_df.groupby(id_col)["missing_periods"]
            .agg(n_gaps="count", total_missing="sum", longest_gap="max")
            .reset_index()
        )

    span = (
        df.groupby(id_col)[ts_col]
        .agg(first_obs="min", last_obs="max", total_observed="count")
        .reset_index()
    )
    span["expected_periods"] = (
        ((span["last_obs"] - span["first_obs"]) / expected).round().astype(int) + 1
    )

    result = span.merge(gap_agg, on=id_col, how="left")
    result[["n_gaps", "total_missing", "longest_gap"]] = (
        result[["n_gaps", "total_missing", "longest_gap"]].fillna(0).astype(int)
    )
    result["missingness_rate"] = result["total_missing"] / result["expected_periods"].clip(lower=1)

    return result.sort_values("total_missing", ascending=False).reset_index(drop=True)


def summarise_gaps(
    df: pd.DataFrame,
    id_col: str = "LCLid",
    ts_col: str = "tstp",
    freq: str = "30min",
    print_summary: bool = True,
) -> dict:
    """
    Return (and optionally print) a dataset-level gap summary.

    Keys in the returned dict:
        n_households            total households
        n_households_with_gaps  households that have at least one gap
        total_gaps              total number of distinct gaps
        total_missing_periods   sum of all missing_periods across every gap
        overall_missingness_rate  total_missing / total_expected
        gap_length_distribution  Series: percentiles of missing_periods (per gap)
    """
    gap_df  = find_gaps(df, id_col=id_col, ts_col=ts_col, freq=freq)
    hh_df   = gaps_per_household(df, id_col=id_col, ts_col=ts_col, freq=freq)

    n_hh           = df[id_col].nunique()
    n_hh_with_gaps = int((hh_df["n_gaps"] > 0).sum())
    total_gaps     = int(gap_df["missing_periods"].count()) if not gap_df.empty else 0
    total_missing  = int(hh_df["total_missing"].sum())
    total_expected = int(hh_df["expected_periods"].sum())
    overall_miss   = total_missing / total_expected if total_expected else 0.0

    pcts = (
        gap_df["missing_periods"].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.99])
        if not gap_df.empty
        else pd.Series(dtype=float)
    )

    summary = {
        "n_households": n_hh,
        "n_households_with_gaps": n_hh_with_gaps,
        "total_gaps": total_gaps,
        "total_missing_periods": total_missing,
        "overall_missingness_rate": overall_miss,
        "gap_length_distribution": pcts,
    }

    if print_summary:
        _print_summary(summary, freq)

    return summary


# ---------------------------------------------------------------------------
# Resampled-grid gap detection (NaN-run approach)
# ---------------------------------------------------------------------------

def label_gap_lengths(series: pd.Series) -> pd.Series:
    """
    Return a Series aligned with *series* where each position holds the length
    of the NaN run it belongs to, or 0 if the original value was observed.
    """
    is_nan = series.isna()
    run_id = (is_nan != is_nan.shift()).cumsum()
    run_sizes = is_nan.groupby(run_id).transform("size")
    return run_sizes.where(is_nan, 0).astype(int)


def gap_length_dataframe(
    hh_df: pd.DataFrame,
    id_col: str = "LCLid",
    value_col: str = "energy(kWh/hh)",
) -> pd.DataFrame:
    """
    Build a flat DataFrame with one row per gap across all households in a
    resampled grid (MultiIndex: id_col × tstp).

    Columns returned:
        id_col      household identifier
        gap_length  number of consecutive missing half-hour slots
    """
    s = hh_df[value_col]
    is_nan = s.isna()
    lclid  = s.index.get_level_values(id_col)

    # A new run starts at any NaN↔non-NaN transition OR a household boundary
    lclid_s  = pd.Series(lclid, index=s.index)
    new_run  = (is_nan != is_nan.shift()) | (lclid_s != lclid_s.shift().fillna(""))
    run_id   = new_run.cumsum()
    run_sizes = is_nan.groupby(run_id).transform("size")

    is_gap_start = is_nan & new_run
    return pd.DataFrame({
        id_col:      lclid[is_gap_start.to_numpy()],
        "gap_length": run_sizes.loc[is_gap_start].to_numpy(),
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal print helper
# ---------------------------------------------------------------------------

def _print_summary(summary: dict, freq: str) -> None:
    pcts = summary["gap_length_distribution"]
    print(f"\n{'='*60}")
    print(f"GAP ANALYSIS  (resolution: {freq})")
    print(f"{'='*60}")
    print(f"  Households total          : {summary['n_households']}")
    print(f"  Households with gaps      : {summary['n_households_with_gaps']}")
    print(f"  Total distinct gaps       : {summary['total_gaps']}")
    print(f"  Total missing periods     : {summary['total_missing_periods']}")
    print(f"  Overall missingness rate  : {summary['overall_missingness_rate']:.2%}")
    if not pcts.empty:
        print(f"\n  Gap-length distribution (periods):")
        for stat in ["min", "25%", "50%", "75%", "90%", "99%", "max"]:
            if stat in pcts.index:
                print(f"    {stat:>5s}  {pcts[stat]:.1f}")
    print(f"{'='*60}\n")

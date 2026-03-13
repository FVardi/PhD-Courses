"""
Gap analysis helpers for time-series data.

Three public functions:
  find_gaps            – raw table of every gap per household
  gaps_per_household   – per-household summary (count, total missing, longest gap)
  summarise_gaps       – dataset-level missingness rate + gap-length distribution
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Core gap detection
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

        # Any delta strictly larger than expected contains a gap
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


# ---------------------------------------------------------------------------
# Per-household summary
# ---------------------------------------------------------------------------

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

    # Aggregate gaps per household
    if gap_df.empty:
        gap_agg = pd.DataFrame(columns=[id_col, "n_gaps", "total_missing", "longest_gap"])
    else:
        gap_agg = (
            gap_df.groupby(id_col)["missing_periods"]
            .agg(n_gaps="count", total_missing="sum", longest_gap="max")
            .reset_index()
        )

    # Compute expected span per household from the actual data
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


# ---------------------------------------------------------------------------
# Dataset-level summary
# ---------------------------------------------------------------------------

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
    expected = pd.Timedelta(freq)
    gap_df   = find_gaps(df, id_col=id_col, ts_col=ts_col, freq=freq)
    hh_df    = gaps_per_household(df, id_col=id_col, ts_col=ts_col, freq=freq)

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

"""
00: Interactive household inspection

Change LCLID (or LCLIDS) and re-run any cell to inspect a different household.
Data is loaded from the raw halfhourly CSV blocks via DataLoader.
"""

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plots import (
    plot_acf_pacf,
    plot_daily_profile,
    plot_halfhourly_zoom,
    plot_households_with_splits,
)

DATA_ROOT     = PROJECT_ROOT / "data" / "london_smart_meters"
IMPUTED_PATH  = PROJECT_ROOT / "data" / "preprocessed" / "halfhourly_imputed.parquet"
VALUE_COL     = "energy(kWh/hh)"
TRAIN_END  = pd.Timestamp("2014-01-01")
TEST_START = pd.Timestamp("2014-02-01")

# %%
# ---- Configure here ----------------------------------------------------------
LCLID  = "MAC000018"          # single household for zoom / ACF plots
LCLIDS = [
    "MAC000018",
    "MAC000049",
    "MAC000255",
    "MAC000313",
    "MAC000855",
    "MAC001150",
    "MAC001367",
    "MAC002166",
    "MAC002600",
    "MAC002796",
    "MAC002918",
    "MAC003199",
    "MAC003242",
    "MAC003356",
    "MAC003388",
    "MAC003389",
    "MAC003417",
    "MAC003431",
    "MAC003436",
    "MAC003445",
]
ACF_LAGS = 336                # 336 = one week of half-hourly data
# ------------------------------------------------------------------------------

# %%
# Load from the imputed parquet — same source as quality screening
_raw = pd.read_parquet(
    IMPUTED_PATH,
    filters=[("LCLid", "in", LCLIDS)],
    columns=[VALUE_COL],
)
df = _raw.reset_index()

_found = df["LCLid"].unique().tolist()
_missing = sorted(set(LCLIDS) - set(_found))
if _missing:
    print(f"WARNING: not found in imputed parquet: {_missing}")
print(f"Loaded {df['LCLid'].nunique()} / {len(LCLIDS)} households, {len(df):,} rows  "
      f"({df['tstp'].min().date()} → {df['tstp'].max().date()})")

# %%
# Build MultiIndex version for split-aware plots
data = df.set_index(["LCLid", "tstp"])[[VALUE_COL]]

# %%
# --- Full time series with train/val/test boundaries -------------------------
fig = plot_households_with_splits(
    data, LCLIDS, VALUE_COL,
    train_end=TRAIN_END, test_start=TEST_START,
)
fig.suptitle("Full series — train / val / test splits", y=1.01)
plt.show()

# %%
# --- Three zoom levels (3 months / 1 week / 1 day) ---------------------------
fig = plot_halfhourly_zoom(df, household_ids=LCLIDS[:5])
plt.show()

# %%
# --- Average daily profile (mean ± std per 30-min slot) ----------------------
fig = plot_daily_profile(df)
plt.show()

# %%
# --- ACF / PACF for a single household ---------------------------------------
fig = plot_acf_pacf(
    df[df["LCLid"] == LCLID],
    household_ids=[LCLID],
    lags=ACF_LAGS,
    timestamp_col="tstp",
    value_col=VALUE_COL,
)
plt.show()

# %%
# --- Quick summary stats (training period) -----------------------------------
series = (
    data.xs(LCLID, level="LCLid")[VALUE_COL]
    .loc[lambda s: s.index < TRAIN_END]
    .dropna()
)
print(f"\n{LCLID} — training period summary")
print(series.describe().round(4).to_string())
print(f"Missing slots : {series.isna().sum()}")
print(f"Zero slots    : {(series == 0).sum()}")

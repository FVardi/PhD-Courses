"""
Task 02: Missing Data Analysis and Imputation

Pipeline:
  1. Load preprocessed halfhourly data and resample to a complete 30-min grid.
  2. Analyse gap structure (distribution, worst-case households).
  3. Classify gaps as short (<7 slots, ≤3 h) or long (≥7 slots).
  4. Impute missing values with two strategies and compare residual NaNs.
  5. Build and save a final annotated dataset with gap_length and is_imputed columns.

Reusable logic lives in:
  src/utils/resampling.py        (build_resampled_grid)
  src/evaluation/gaps.py         (gap_length_dataframe, label_gap_lengths)
  src/imputation/strategies.py   (naive_impute, seasonal_mean_impute)
"""

# %%
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import DataLoader
from src.utils.resampling import build_resampled_grid
from src.evaluation.gaps import gap_length_dataframe, label_gap_lengths
from src.imputation.strategies import naive_impute, seasonal_mean_impute

# %%

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_ROOT   = PROJECT_ROOT / "data" / "london_smart_meters"
OUTPUT_DIR  = PROJECT_ROOT / "data" / "preprocessed"
VALUE_COL   = "energy(kWh/hh)"

# ── Run mode ──────────────────────────────────────────────────────────────────
# QUICK_RUN=True  → load only block_0 (~500 households); fast for inspection.
# QUICK_RUN=False → load all blocks; required before running tasks 03–10.
QUICK_RUN = True
BLOCKS    = [0] if QUICK_RUN else None

# %%
# --- Step 1: Load and resample -------------------------------------------------

logger.info("Loading halfhourly_dataset (blocks %s) …", BLOCKS)
loader = DataLoader(data_root=DATA_ROOT)
raw, meta = loader.load_and_clean_dataset(
    dataset_type="halfhourly_dataset", blocks=BLOCKS, sort=True
)
logger.info(
    "Loaded %d rows across %d households (%d duplicates removed, %d invalid records)",
    meta["final_rows"], raw["LCLid"].nunique(),
    meta["duplicates_removed"], meta["invalid_records"],
)

hh_df = build_resampled_grid(raw)
n_slots = len(hh_df)
logger.info(
    "Resampled to complete 30-min grid: %d total slots across %d households",
    n_slots,
    hh_df.index.get_level_values("LCLid").nunique(),
)

# %%
# --- Step 2: Gap structure analysis -------------------------------------------

initial_nan = int(hh_df[VALUE_COL].isna().sum())
logger.info(
    "Missing slots: %d / %d  (%.2f%%)",
    initial_nan, n_slots, initial_nan / n_slots * 100,
)

gap_df = gap_length_dataframe(hh_df)
logger.info(
    "Gap structure: %d distinct gaps across %d households",
    len(gap_df), gap_df["LCLid"].nunique(),
)
logger.info("Gap length distribution (half-hour slots):\n%s", gap_df["gap_length"].describe())

# Worst-case households
total_per_hh   = hh_df.groupby(level="LCLid")[VALUE_COL].size()
missing_per_hh = hh_df.groupby(level="LCLid")[VALUE_COL].apply(lambda s: s.isna().sum())
missingness_pct = (missing_per_hh / total_per_hh * 100).rename("missingness_pct")
longest_gap     = gap_df.groupby("LCLid")["gap_length"].max().rename("longest_gap")
household_stats = pd.concat([missingness_pct, longest_gap], axis=1).fillna(0)

N = 3
logger.info("Top %d households by missingness %%:\n%s", N, household_stats.nlargest(N, "missingness_pct"))
logger.info("Top %d households by longest gap:\n%s",    N, household_stats.nlargest(N, "longest_gap"))

# %%
# --- Step 3: Household quality filter -----------------------------------------
#
# Drop households whose training-period data does not meet quality thresholds.
# Thresholds are evaluated on the full loaded series (no train/test split yet).

MAX_MISSINGNESS_PCT = 10.0   # drop if > 5 % of slots are missing
MAX_GAP_SLOTS       = 48    # drop if longest single gap exceeds 1 day (48 half-hours)

fail_missingness = household_stats[household_stats["missingness_pct"] > MAX_MISSINGNESS_PCT].index
fail_gap         = household_stats[household_stats["longest_gap"]     > MAX_GAP_SLOTS].index
discard          = fail_missingness.union(fail_gap)

logger.info(
    "Quality filter: %d households discarded  "
    "(%d exceed missingness >%.0f%%, %d exceed max gap >%d slots, overlap counted once)",
    len(discard), len(fail_missingness), MAX_MISSINGNESS_PCT, len(fail_gap), MAX_GAP_SLOTS,
)
if len(discard):
    logger.info("Discarded households:\n%s", household_stats.loc[discard].to_string())

hh_df = hh_df[~hh_df.index.get_level_values("LCLid").isin(discard)].copy()
logger.info(
    "Retained %d households after quality filter",
    hh_df.index.get_level_values("LCLid").nunique(),
)

# Recompute slot count and initial NaN count on the filtered frame
n_slots     = len(hh_df)
initial_nan = int(hh_df[VALUE_COL].isna().sum())

# %%
# --- Step 4: Short / long gap classification (on filtered households) ---------
#
# gap_df is rebuilt from the filtered hh_df so the threshold decision and
# logged statistics reflect only the households that will be imputed and saved.

gap_df = gap_length_dataframe(hh_df)
logger.info(
    "Gap structure after filter: %d distinct gaps across %d households",
    len(gap_df), gap_df["LCLid"].nunique(),
)
logger.info("Gap length distribution after filter:\n%s", gap_df["gap_length"].describe())

# Decision: threshold = 9 half-hour slots.
# The 75th percentile of all gap lengths is 6.75 slots, meaning three-quarters
# of gaps are 6 slots or fewer.  Setting the boundary at 9 captures this natural
# majority as "short" (safely interpolable from local context) while flagging the
# heavier-tailed minority as "long" (structural outages requiring special handling).

SHORT_THRESHOLD = 2
short_count = int((gap_df["gap_length"] <  SHORT_THRESHOLD).sum())
long_count  = int((gap_df["gap_length"] >= SHORT_THRESHOLD).sum())
logger.info(
    "Gap classification (threshold=%d slots = %.1f h): short=%d, long=%d",
    SHORT_THRESHOLD, SHORT_THRESHOLD * 0.5, short_count, long_count,
)

# %%
# --- Step 5: Imputation -------------------------------------------------------
#
# Strategy A — Naive lag-48: fill each NaN with the value from the identical
#   half-hour slot 24 hours earlier.  Fast and interpretable but fails when the
#   lag slot is also missing.
#
# Strategy B — Seasonal mean (primary): fill each NaN with the causal expanding
#   mean of all previous observations at that half-hour slot for the same
#   household.  Uses no future data and improves as history accumulates.  This
#   is the strategy used for the saved output.
# TODO: Make it so that both imputation methods are saved in the output dataset, so we can compare them in later tasks.
logger.info("Running naive (lag-48) imputation …")
imputed_naive = (
    hh_df.groupby(level="LCLid")[VALUE_COL]
    .transform(naive_impute)
)
remaining_naive = int(imputed_naive.isna().sum())
filled_naive    = initial_nan - remaining_naive
logger.info(
    "Naive imputation:          filled %d / %d NaNs  (%d remaining)",
    filled_naive, initial_nan, remaining_naive,
)

logger.info("Running seasonal mean imputation …")
imputed_seasonal = seasonal_mean_impute(hh_df, value_col=VALUE_COL)
remaining_seasonal = int(imputed_seasonal.isna().sum())
filled_seasonal    = initial_nan - remaining_seasonal
logger.info(
    "Seasonal mean imputation:  filled %d / %d NaNs  (%d remaining)",
    filled_seasonal, initial_nan, remaining_seasonal,
)

logger.info(
    "Selected strategy for output: seasonal mean  "
    "(fills %d more NaNs than naive, more robust for long gaps)",
    filled_seasonal - filled_naive,
)

# %%
# --- Step 6: Build and save annotated output ----------------------------------
#
# Final DataFrame columns:
#   energy(kWh/hh)            – original observed values (NaN where missing)
#   energy_imputed_seasonal   – seasonally imputed values
#   energy_imputed_naive      – naive lag-48 imputed values
#   gap_length                – length of the NaN run the slot belonged to (0 if observed)
#   is_imputed                – True where the original was missing

logger.info("Building annotated output dataset …")

gap_length_col = (
    hh_df.groupby(level="LCLid")[VALUE_COL]
    .transform(label_gap_lengths)
    .rename("gap_length")
)

result = hh_df.copy()
result["energy_imputed_seasonal"] = imputed_seasonal
result["energy_imputed_naive"]    = imputed_naive
result["gap_length"]              = gap_length_col
result["is_imputed"]              = hh_df[VALUE_COL].isna()

out_path = OUTPUT_DIR / "halfhourly_imputed.parquet"
result.to_parquet(out_path)
logger.info(
    "Saved annotated dataset → %s  (%.1f MB, %d rows, %d imputed slots)",
    out_path,
    out_path.stat().st_size / 1e6,
    len(result),
    int(result["is_imputed"].sum()),
)

# %%

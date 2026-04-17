"""
Task 02: Missing Data Analysis
"""

# %%
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# %%

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

# %%
# Load a small slice: first N_HOUSEHOLDS from halfhourly_dataset

hh_path = PREPROCESSED_DIR / "halfhourly_dataset.parquet"
logger.info("Loading %s", hh_path)

raw = pd.read_parquet(hh_path, columns=["LCLid", "tstp", "energy(kWh/hh)"])

logger.info("Loaded %d rows for %d households", len(raw), raw["LCLid"].nunique())

# %%
# Resample each household to a complete 30-min grid, introducing NaN where data is missing. Calculate percentage missingness.

raw["tstp"] = pd.to_datetime(raw["tstp"])
raw = raw.set_index("tstp")

def resample_household(df: pd.DataFrame) -> pd.DataFrame:
    return df[["energy(kWh/hh)"]].resample("30min").mean()

hh_df = (
    raw.groupby("LCLid")
    .apply(resample_household)
)

logger.info("Resampled shape: %s", hh_df.shape)

# %%
# For each household, find consecutive NaN runs and record their lengths

def find_gap_lengths(series: pd.Series) -> list[int]:
    is_nan = series.isna()
    run_id = (is_nan != is_nan.shift()).cumsum()
    return [
        len(group)
        for _, group in is_nan.groupby(run_id)
        if group.iloc[0]  # only NaN runs
    ]

records = []
for lclid, group in hh_df.groupby(level="LCLid"):
    for length in find_gap_lengths(group["energy(kWh/hh)"]):
        records.append({"LCLid": lclid, "gap_length": length})

gap_df = pd.DataFrame(records)

logger.info("Total gaps found: %d across %d households", len(gap_df), gap_df["LCLid"].nunique())
logger.info("Gap length statistics across all households:\n%s", gap_df["gap_length"].describe())

# %%
# Worst cases: by missingness % and by longest single gap

total_per_hh = hh_df.groupby(level="LCLid")["energy(kWh/hh)"].size()
missing_per_hh = hh_df.groupby(level="LCLid")["energy(kWh/hh)"].apply(lambda s: s.isna().sum())
missingness_pct = (missing_per_hh / total_per_hh * 100).rename("missingness_pct")

longest_gap = gap_df.groupby("LCLid")["gap_length"].max().rename("longest_gap")

household_stats = pd.concat([missingness_pct, longest_gap], axis=1).fillna(0)

N = 3
logger.info("Top %d by missingness %%:\n%s", N, household_stats.nlargest(N, "missingness_pct"))
logger.info("Top %d by longest gap:\n%s", N, household_stats.nlargest(N, "longest_gap"))

# %%
SHORT_THRESHOLD = 7
short = (gap_df["gap_length"] < SHORT_THRESHOLD).sum()
long  = (gap_df["gap_length"] >= SHORT_THRESHOLD).sum()
logger.info("Short gaps (<7): %d  |  Long gaps (>=7): %d", short, long)

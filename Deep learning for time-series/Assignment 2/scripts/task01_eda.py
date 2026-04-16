"""
Task 1 — Data Loading, Integrity Checks, and Exploratory Analysis
==================================================================
Subtask 1: Load and validate the schema.

  - Validate schema and dtypes from parquet metadata (no full load needed).
  - Verify column presence, timestamp parsing, sort order, and key uniqueness
    using a household-level sample to avoid loading 167M rows into memory.
  - Print a full validation report and dataset overview.

Reusable utilities: src/utils/data_loader.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import pandas as pd
import pyarrow.parquet as pq

from src.utils.data_loader import (
    COL_HOUSEHOLD_ID,
    COL_TARGET,
    COL_TIMESTAMP,
    SCHEMA,
    SCHEMA_META,
)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
FULL_PATH  = os.path.join(PROCESSED_DIR, "full_dataset.parquet")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.parquet")
VAL_PATH   = os.path.join(PROCESSED_DIR, "val.parquet")
TEST_PATH  = os.path.join(PROCESSED_DIR, "test.parquet")

# Number of households to load for data quality spot-checks
SAMPLE_N_HOUSEHOLDS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parquet_schema_and_counts(path: str) -> tuple[pq.ParquetSchema, int]:
    """Read parquet metadata only — no data loaded."""
    meta = pq.read_metadata(path)
    schema = pq.read_schema(path)
    n_rows = meta.num_rows
    return schema, n_rows


def load_sample(path: str, n_households: int = SAMPLE_N_HOUSEHOLDS) -> pd.DataFrame:
    """
    Load a sample of households from a parquet file efficiently.
    Reads all rows but only the key columns needed for QC, then filters
    to the first n_households unique LCLids.
    """
    df = pd.read_parquet(path, columns=[COL_HOUSEHOLD_ID, COL_TIMESTAMP, COL_TARGET])
    ids = df[COL_HOUSEHOLD_ID].unique()[:n_households]
    return df[df[COL_HOUSEHOLD_ID].isin(ids)].copy()


# ---------------------------------------------------------------------------
# Subtask 1 — Schema validation
# ---------------------------------------------------------------------------
def subtask1_load_and_validate() -> None:
    print("=" * 60)
    print("  Task 1, Subtask 1 — Schema Loading and Validation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Schema from parquet metadata (zero memory cost)
    # ------------------------------------------------------------------
    print("\n=== Parquet Schema (from file metadata) ===")
    arrow_schema, n_rows_full = parquet_schema_and_counts(FULL_PATH)

    expected_schema = {**SCHEMA, **SCHEMA_META}
    all_present = True
    for col in expected_schema:
        if col in arrow_schema.names:
            actual = str(arrow_schema.field(col).type)
            print(f"  [OK]   {col:<25} arrow_type={actual}")
        else:
            print(f"  [FAIL] {col:<25} MISSING")
            all_present = False

    if all_present:
        print("  => All required columns present in parquet schema")
    else:
        print("  => Some columns are missing — check setup.py output")

    # ------------------------------------------------------------------
    # 2. Row counts per split (from metadata only)
    # ------------------------------------------------------------------
    print("\n=== Row Counts per Split (from parquet metadata) ===")
    _, n_train = parquet_schema_and_counts(TRAIN_PATH)
    _, n_val   = parquet_schema_and_counts(VAL_PATH)
    _, n_test  = parquet_schema_and_counts(TEST_PATH)

    print(f"  Full dataset : {n_rows_full:>15,}")
    print(f"  Train        : {n_train:>15,}")
    print(f"  Validation   : {n_val:>15,}")
    print(f"  Test         : {n_test:>15,}")
    check = n_train + n_val + n_test
    if check == n_rows_full:
        print(f"  [OK]   Train + Val + Test = {check:,} (matches full dataset)")
    else:
        print(f"  [WARN] Train+Val+Test={check:,} != Full={n_rows_full:,} "
              f"(rows outside split boundaries)")

    # ------------------------------------------------------------------
    # 3. Load a sample for data-quality checks
    # ------------------------------------------------------------------
    print(f"\n=== Data Quality Checks (sample: {SAMPLE_N_HOUSEHOLDS} households) ===")
    sample = load_sample(FULL_PATH, SAMPLE_N_HOUSEHOLDS)
    actual_n = sample[COL_HOUSEHOLD_ID].nunique()
    print(f"  Loaded {len(sample):,} rows across {actual_n} households")

    # Timestamp dtype
    ts_dtype = sample[COL_TIMESTAMP].dtype
    if pd.api.types.is_datetime64_any_dtype(ts_dtype):
        print(f"  [OK]   {COL_TIMESTAMP} dtype: {ts_dtype}")
    else:
        print(f"  [FAIL] {COL_TIMESTAMP} dtype: {ts_dtype} (expected datetime)")

    # Target dtype and null rate
    tgt_dtype = sample[COL_TARGET].dtype
    n_null = sample[COL_TARGET].isna().sum()
    pct_null = 100 * n_null / len(sample)
    if pd.api.types.is_float_dtype(tgt_dtype):
        print(f"  [OK]   {COL_TARGET} dtype: {tgt_dtype}  nulls: {n_null} ({pct_null:.2f}%)")
    else:
        print(f"  [FAIL] {COL_TARGET} dtype: {tgt_dtype} (expected float64)")

    # Sort order — per household timestamps must be strictly ascending
    monotone = (
        sample.groupby(COL_HOUSEHOLD_ID)[COL_TIMESTAMP]
        .apply(lambda s: (s.diff().dropna() > pd.Timedelta(0)).all())
    )
    n_ok = monotone.sum()
    print(f"  [{'OK' if n_ok == actual_n else 'WARN'}]   "
          f"Strictly ascending timestamps: {n_ok}/{actual_n} households")

    # Key uniqueness — no duplicate (household, timestamp) pairs
    n_dups = sample.duplicated(subset=[COL_HOUSEHOLD_ID, COL_TIMESTAMP]).sum()
    print(f"  [{'OK' if n_dups == 0 else 'FAIL'}]   "
          f"Duplicate (household, timestamp) keys: {n_dups}")

    # Timestamp spacing — should be 30-minute intervals
    diffs = (
        sample.groupby(COL_HOUSEHOLD_ID)[COL_TIMESTAMP]
        .apply(lambda s: s.diff().dropna())
    )
    freq_counts = diffs.value_counts().head(5)
    print(f"  Timestamp gap distribution (top 5):")
    for gap, count in freq_counts.items():
        print(f"    {str(gap):<20} : {count:,} occurrences")

    # ------------------------------------------------------------------
    # 4. Dataset overview from sample
    # ------------------------------------------------------------------
    print("\n=== Dataset Overview ===")
    full_meta = pq.read_metadata(FULL_PATH)
    ts_sample = sample[COL_TIMESTAMP]
    print(f"  Total rows (full)   : {n_rows_full:,}")
    print(f"  Parquet row groups  : {full_meta.num_row_groups}")
    print(f"  Timestamp range     : {ts_sample.min()} to {ts_sample.max()} (from sample)")
    print(f"  30-min slots per day: 48")
    print(f"  Columns             : {arrow_schema.names}")

    print("\nSubtask 1 complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    subtask1_load_and_validate()

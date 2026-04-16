"""
Data Bundle Setup and Experimental Split
=========================================
Loads the London Smart Meters HALF-HOURLY dataset from all 112 block files,
joins household metadata, validates the schema, applies the
train/validation/test split, and writes processed outputs to data/processed/.

Memory strategy:
    Blocks are processed one at a time. Each block is parsed, split, and
    streamed into the output parquet files via PyArrow ParquetWriter so that
    the full dataset is never held in memory simultaneously.

Column mapping:
    household_id : LCLid
    timestamp    : tstp  (half-hourly, 30-minute intervals)
    target       : energy_kWh_per_hh  (renamed from 'energy(kWh/hh)')
    meta_label   : Acorn_grouped
    other meta   : stdorToU, Acorn

Missing values:
    The raw target uses the string "Null" as a sentinel. Replaced with NaN.

Split:
    Train      : tstp < 2014-01-01
    Validation : 2014-01-01 <= tstp <= 2014-01-31 23:30
    Test       : 2014-02-01 <= tstp <= 2014-02-28 23:30
"""

import glob
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_DIR, "data", "london_smart_meters")
BLOCKS_DIR = os.path.join(DATA_RAW, "halfhourly_dataset", "halfhourly_dataset")
DATA_PROCESSED = os.path.join(PROJECT_DIR, "data", "processed")
os.makedirs(DATA_PROCESSED, exist_ok=True)

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
COL_HOUSEHOLD_ID = "LCLid"
COL_TIMESTAMP    = "tstp"
COL_TARGET       = "energy_kWh_per_hh"   # renamed — parens invalid in queries
COL_TARGET_RAW   = "energy(kWh/hh)"
COL_META_LABEL   = "Acorn_grouped"
COL_META_ACORN   = "Acorn"
COL_META_STD_TOU = "stdorToU"

# Split boundaries
TRAIN_END  = pd.Timestamp("2013-12-31 23:30:00")
VAL_START  = pd.Timestamp("2014-01-01 00:00:00")
VAL_END    = pd.Timestamp("2014-01-31 23:30:00")
TEST_START = pd.Timestamp("2014-02-01 00:00:00")
TEST_END   = pd.Timestamp("2014-02-28 23:30:00")

# Output file names
OUT_FULL  = os.path.join(DATA_PROCESSED, "full_dataset.parquet")
OUT_TRAIN = os.path.join(DATA_PROCESSED, "train.parquet")
OUT_VAL   = os.path.join(DATA_PROCESSED, "val.parquet")
OUT_TEST  = os.path.join(DATA_PROCESSED, "test.parquet")


# ---------------------------------------------------------------------------
# Load household metadata (small — fine to hold in memory)
# ---------------------------------------------------------------------------
def load_metadata(data_raw: str) -> pd.DataFrame:
    path = os.path.join(data_raw, "informations_households.csv")
    meta = pd.read_csv(
        path,
        usecols=[COL_HOUSEHOLD_ID, COL_META_STD_TOU, COL_META_ACORN, COL_META_LABEL],
    )
    return meta.drop_duplicates(subset=COL_HOUSEHOLD_ID)


# ---------------------------------------------------------------------------
# Parse a single block file
# ---------------------------------------------------------------------------
def parse_block(path: str, meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={COL_TARGET_RAW: "str"}, low_memory=False)

    # Rename target
    df = df.rename(columns={COL_TARGET_RAW: COL_TARGET})

    # Parse timestamp
    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], utc=False)

    # Cast target: "Null" sentinel -> float NaN
    df[COL_TARGET] = pd.to_numeric(df[COL_TARGET], errors="coerce")

    # Sort within block
    df = df.sort_values([COL_HOUSEHOLD_ID, COL_TIMESTAMP])

    # Remove duplicates within block
    df = df.drop_duplicates(subset=[COL_HOUSEHOLD_ID, COL_TIMESTAMP], keep="first")

    # Join metadata
    df = df.merge(meta, on=COL_HOUSEHOLD_ID, how="left")

    return df


# ---------------------------------------------------------------------------
# Stream-write all blocks into split parquet files
# ---------------------------------------------------------------------------
def process_blocks(block_files: list[str], meta: pd.DataFrame) -> dict:
    writers: dict[str, pq.ParquetWriter] = {}
    schema = None

    stats = {
        "total_rows": 0,
        "train_rows": 0,
        "val_rows": 0,
        "test_rows": 0,
        "null_target": 0,
        "duplicates_removed": 0,
    }

    for i, path in enumerate(block_files, 1):
        df = parse_block(path, meta)

        stats["total_rows"] += len(df)
        stats["null_target"] += df[COL_TARGET].isna().sum()

        # Build PyArrow schema from first block
        table_full = pa.Table.from_pandas(df, preserve_index=False)
        if schema is None:
            schema = table_full.schema
            writers["full"]  = pq.ParquetWriter(OUT_FULL,  schema)
            writers["train"] = pq.ParquetWriter(OUT_TRAIN, schema)
            writers["val"]   = pq.ParquetWriter(OUT_VAL,   schema)
            writers["test"]  = pq.ParquetWriter(OUT_TEST,  schema)

        writers["full"].write_table(table_full)

        # Split
        train_df = df[df[COL_TIMESTAMP] <= TRAIN_END]
        val_df   = df[(df[COL_TIMESTAMP] >= VAL_START) & (df[COL_TIMESTAMP] <= VAL_END)]
        test_df  = df[(df[COL_TIMESTAMP] >= TEST_START) & (df[COL_TIMESTAMP] <= TEST_END)]

        stats["train_rows"] += len(train_df)
        stats["val_rows"]   += len(val_df)
        stats["test_rows"]  += len(test_df)

        if len(train_df): writers["train"].write_table(pa.Table.from_pandas(train_df, schema=schema, preserve_index=False))
        if len(val_df):   writers["val"].write_table(pa.Table.from_pandas(val_df,   schema=schema, preserve_index=False))
        if len(test_df):  writers["test"].write_table(pa.Table.from_pandas(test_df,  schema=schema, preserve_index=False))

        if i % 20 == 0 or i == len(block_files):
            print(f"  [{i}/{len(block_files)}] processed  —  {stats['total_rows']:,} rows so far")

    for w in writers.values():
        w.close()

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  London Smart Meters — Setup & Experimental Split")
    print("  (Half-hourly dataset, streaming block-by-block)")
    print("=" * 60)

    block_files = sorted(glob.glob(os.path.join(BLOCKS_DIR, "block_*.csv")))
    if not block_files:
        sys.exit(f"ERROR: No block files found in {BLOCKS_DIR}")
    print(f"Found {len(block_files)} block files")

    print("Loading household metadata...")
    meta = load_metadata(DATA_RAW)
    print(f"  {len(meta):,} households in metadata")

    print("\nProcessing blocks (streaming to parquet)...")
    stats = process_blocks(block_files, meta)

    print("\n=== Results ===")
    print(f"  Total rows          : {stats['total_rows']:,}")
    print(f"  Target nulls (Null) : {stats['null_target']:,}  "
          f"({100*stats['null_target']/stats['total_rows']:.2f}%)")
    print(f"\n  Train rows          : {stats['train_rows']:,}")
    print(f"  Validation rows     : {stats['val_rows']:,}")
    print(f"  Test rows           : {stats['test_rows']:,}")

    print(f"\n=== Saved to {DATA_PROCESSED} ===")
    for path in (OUT_FULL, OUT_TRAIN, OUT_VAL, OUT_TEST):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {os.path.basename(path):<30} ({size_mb:,.1f} MB)")

    print("\nColumn reference for downstream tasks:")
    print(f"  household_id : {COL_HOUSEHOLD_ID}")
    print(f"  timestamp    : {COL_TIMESTAMP}  (30-minute intervals)")
    print(f"  target       : {COL_TARGET}  (kWh per half-hour)")
    print(f"  meta_label   : {COL_META_LABEL}")
    print(f"  other meta   : {COL_META_STD_TOU}, {COL_META_ACORN}")
    print("\nDone.")


if __name__ == "__main__":
    main()

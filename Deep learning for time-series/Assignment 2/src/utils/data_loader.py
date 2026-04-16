"""
Data loading and schema validation utilities.

Provides strict schema enforcement for the London Smart Meters half-hourly dataset:
  - Expected columns with required dtypes
  - Timestamp parsing
  - Sort order enforcement (household_id, timestamp)
  - Duplicate key detection and resolution
  - Schema validation reporting
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Column constants (single source of truth for the whole project)
# ---------------------------------------------------------------------------
COL_HOUSEHOLD_ID = "LCLid"
COL_TIMESTAMP = "tstp"
COL_TARGET = "energy_kWh_per_hh"   # kWh per half-hour slot
COL_META_LABEL = "Acorn_grouped"
COL_META_ACORN = "Acorn"
COL_META_STD_TOU = "stdorToU"

# Expected dtypes for the processed parquet files
SCHEMA: dict[str, str] = {
    COL_HOUSEHOLD_ID: "object",
    COL_TIMESTAMP:    "datetime64[ns]",
    COL_TARGET:       "float64",
}

SCHEMA_META: dict[str, str] = {
    COL_META_LABEL:   "object",
    COL_META_ACORN:   "object",
    COL_META_STD_TOU: "object",
}


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------
@dataclass
class SchemaReport:
    """Collects all findings from schema validation."""
    missing_columns: list[str] = field(default_factory=list)
    wrong_dtype: dict[str, tuple[str, str]] = field(default_factory=dict)
    n_duplicates_removed: int = 0
    duplicate_examples: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_rows_before: int = 0
    n_rows_after: int = 0
    sort_applied: bool = False

    def is_valid(self) -> bool:
        return not self.missing_columns and not self.wrong_dtype

    def print(self) -> None:
        print("\n=== Schema Validation Report ===")
        print(f"  Rows before validation : {self.n_rows_before:,}")
        print(f"  Rows after  validation : {self.n_rows_after:,}")

        if self.missing_columns:
            print(f"  [FAIL] Missing columns : {self.missing_columns}")
        else:
            print(f"  [OK]   All required columns present")

        if self.wrong_dtype:
            for col, (actual, expected) in self.wrong_dtype.items():
                print(f"  [WARN] Column '{col}': dtype={actual}, expected={expected} (cast applied)")
        else:
            print(f"  [OK]   All dtypes correct")

        if self.n_duplicates_removed:
            print(f"  [WARN] Removed {self.n_duplicates_removed:,} duplicate (household, timestamp) rows")
            if not self.duplicate_examples.empty:
                print("         Example duplicates:")
                print(self.duplicate_examples.to_string(index=False))
        else:
            print(f"  [OK]   No duplicate (household_id, timestamp) keys")

        if self.sort_applied:
            print(f"  [OK]   Sorted by ({COL_HOUSEHOLD_ID}, {COL_TIMESTAMP})")


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------
def load_dataset(
    path: str,
    validate: bool = True,
    include_meta: bool = True,
) -> tuple[pd.DataFrame, SchemaReport]:
    """
    Load a processed parquet file with strict schema enforcement.

    Steps performed:
      1. Read parquet file.
      2. Check all required columns are present.
      3. Cast columns to expected dtypes (timestamp parsing included).
      4. Sort by (household_id, timestamp).
      5. Detect and remove duplicate (household_id, timestamp) keys.

    Parameters
    ----------
    path : str
        Path to a processed parquet file (train/val/test/full).
    validate : bool
        Whether to run schema checks. Default True.
    include_meta : bool
        Whether to include metadata columns in dtype checks. Default True.

    Returns
    -------
    df : pd.DataFrame
    report : SchemaReport
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_parquet(path)
    report = SchemaReport(n_rows_before=len(df))

    if not validate:
        report.n_rows_after = len(df)
        return df, report

    expected_schema = {**SCHEMA, **(SCHEMA_META if include_meta else {})}

    # 1. Check required columns
    report.missing_columns = [c for c in expected_schema if c not in df.columns]
    if report.missing_columns:
        report.n_rows_after = len(df)
        report.print()
        raise ValueError(f"Dataset is missing required columns: {report.missing_columns}")

    # 2. Cast / enforce dtypes
    for col, expected_dtype in expected_schema.items():
        if col not in df.columns:
            continue
        actual_dtype = str(df[col].dtype)
        if expected_dtype == "datetime64[ns]":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
                report.wrong_dtype[col] = (actual_dtype, expected_dtype)
        elif expected_dtype == "float64":
            if not pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                report.wrong_dtype[col] = (actual_dtype, expected_dtype)
        elif expected_dtype == "object":
            if not pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype("object")
                report.wrong_dtype[col] = (actual_dtype, expected_dtype)

    # 3. Sort by (household_id, timestamp)
    df = df.sort_values([COL_HOUSEHOLD_ID, COL_TIMESTAMP]).reset_index(drop=True)
    report.sort_applied = True

    # 4. Detect and remove duplicates
    duplicate_mask = df.duplicated(subset=[COL_HOUSEHOLD_ID, COL_TIMESTAMP], keep=False)
    n_dup = duplicate_mask.sum()
    if n_dup:
        report.duplicate_examples = (
            df[duplicate_mask]
            .head(10)[[COL_HOUSEHOLD_ID, COL_TIMESTAMP, COL_TARGET]]
        )
        n_before = len(df)
        df = df.drop_duplicates(subset=[COL_HOUSEHOLD_ID, COL_TIMESTAMP], keep="first")
        report.n_duplicates_removed = n_before - len(df)

    report.n_rows_after = len(df)
    return df, report


# ---------------------------------------------------------------------------
# Convenience loaders for each split
# ---------------------------------------------------------------------------
def _resolve_processed_dir(processed_dir: Optional[str]) -> str:
    if processed_dir is not None:
        return processed_dir
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(project_root, "data", "processed")


def load_train(processed_dir: Optional[str] = None, **kwargs) -> tuple[pd.DataFrame, SchemaReport]:
    return load_dataset(os.path.join(_resolve_processed_dir(processed_dir), "train.parquet"), **kwargs)


def load_val(processed_dir: Optional[str] = None, **kwargs) -> tuple[pd.DataFrame, SchemaReport]:
    return load_dataset(os.path.join(_resolve_processed_dir(processed_dir), "val.parquet"), **kwargs)


def load_test(processed_dir: Optional[str] = None, **kwargs) -> tuple[pd.DataFrame, SchemaReport]:
    return load_dataset(os.path.join(_resolve_processed_dir(processed_dir), "test.parquet"), **kwargs)


def load_full(processed_dir: Optional[str] = None, **kwargs) -> tuple[pd.DataFrame, SchemaReport]:
    return load_dataset(os.path.join(_resolve_processed_dir(processed_dir), "full_dataset.parquet"), **kwargs)

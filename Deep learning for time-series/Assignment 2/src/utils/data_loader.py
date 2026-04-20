"""
Reusable data loading and schema validation utilities for the London Smart Meters dataset.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates DataFrames against strict schemas and checks data types."""

    SCHEMAS: Dict[str, Dict[str, str]] = {
        "daily_dataset": {
            "LCLid": "string",
            "day": "datetime64[ns]",
            "energy_median": "float64",
            "energy_mean": "float64",
            "energy_max": "float64",
            "energy_count": "int64",
            "energy_std": "float64",
            "energy_sum": "float64",
            "energy_min": "float64",
        },
        "halfhourly_dataset": {
            "LCLid": "string",
            "tstp": "datetime64[ns]",
            "energy(kWh/hh)": "float64",
        },
        "hhblock_dataset": {
            "LCLid": "string",
            "day": "datetime64[ns]",
            # Plus hh_0 … hh_47 checked separately
        },
    }

    @staticmethod
    def validate_schema(df: pd.DataFrame, schema_type: str) -> Tuple[bool, List[str]]:
        """Return (is_valid, errors) checking required columns exist."""
        errors: List[str] = []
        schema = SchemaValidator.SCHEMAS.get(schema_type, {})

        for col in schema:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if schema_type == "hhblock_dataset":
            for i in range(48):
                col = f"hh_{i}"
                if col not in df.columns:
                    errors.append(f"Missing half-hour column: {col}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_data_types(df: pd.DataFrame, schema_type: str) -> Tuple[bool, List[str]]:
        """Return (is_valid, errors) checking column dtypes match the schema."""
        errors: List[str] = []
        schema = SchemaValidator.SCHEMAS.get(schema_type, {})

        for col, expected in schema.items():
            if col not in df.columns:
                continue
            if "datetime" in expected:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    errors.append(f"{col}: expected datetime64, got {df[col].dtype}")
            elif "int" in expected:
                if not pd.api.types.is_integer_dtype(df[col]):
                    errors.append(f"{col}: expected int, got {df[col].dtype}")
            elif "float" in expected:
                if not pd.api.types.is_float_dtype(df[col]):
                    errors.append(f"{col}: expected float, got {df[col].dtype}")

        return len(errors) == 0, errors


class DataLoader:
    """Load and clean London Smart Meters blocks with schema validation."""

    def __init__(self, data_root: Path, n_jobs: int = 8):
        self.data_root = Path(data_root)
        self.n_jobs    = n_jobs
        self.validator = SchemaValidator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_block(self, dataset_type: str, block_num: int) -> Optional[pd.DataFrame]:
        """Load a single block CSV, parse timestamps, and validate schema."""
        block_path = self.data_root / dataset_type / dataset_type / f"block_{block_num}.csv"

        if not block_path.exists():
            logger.error("File not found: %s", block_path)
            return None

        # Force energy column to str so mixed-type values (e.g. "Null", "") are
        # handled uniformly by _parse_timestamps rather than causing DtypeWarnings.
        _DTYPE_OVERRIDES = {
            "halfhourly_dataset": {"energy(kWh/hh)": str},
        }
        try:
            df = pd.read_csv(
                block_path,
                dtype=_DTYPE_OVERRIDES.get(dataset_type),
                low_memory=False,
            )
        except Exception as exc:
            logger.error("Error reading %s: %s", block_path, exc)
            return None

        df = self._parse_timestamps(df, dataset_type)

        schema_ok, errs = self.validator.validate_schema(df, dataset_type)
        if not schema_ok:
            logger.warning("Schema errors in %s/block_%d: %s", dataset_type, block_num, errs)

        dtype_ok, errs = self.validator.validate_data_types(df, dataset_type)
        if not dtype_ok:
            logger.warning("Dtype errors in %s/block_%d: %s", dataset_type, block_num, errs)

        return df

    def load_and_clean_dataset(
        self,
        dataset_type: str,
        blocks: Optional[List[int]] = None,
        sort: bool = True,
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Load one or more blocks, validate, deduplicate, and optionally sort.

        Args:
            dataset_type:  'daily_dataset', 'halfhourly_dataset', or 'hhblock_dataset'
            blocks:        Block numbers to load; None loads all 112.
            sort:          Sort by (LCLid, timestamp) if True.

        Returns:
            (DataFrame, metadata_dict)
        """
        if blocks is None:
            blocks = list(range(112))

        logger.info("Loading %s (%d blocks)…", dataset_type, len(blocks))

        metadata: Dict = {
            "dataset_type": dataset_type,
            "blocks_loaded": [],
            "blocks_failed": [],
            "total_rows_loaded": 0,
            "duplicates_found": 0,
            "duplicates_removed": 0,
            "invalid_records": 0,
        }

        dfs = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            futures = {pool.submit(self.load_block, dataset_type, b): b for b in blocks}
            for future in as_completed(futures):
                block_num = futures[future]
                df = future.result()
                if df is not None:
                    dfs.append(df)
                    metadata["blocks_loaded"].append(block_num)
                    metadata["total_rows_loaded"] += len(df)
                else:
                    metadata["blocks_failed"].append(block_num)

        if not dfs:
            logger.error("No blocks loaded from %s", dataset_type)
            return None, metadata

        combined = pd.concat(dfs, ignore_index=True)
        logger.info("Combined shape: %s", combined.shape)

        ts_col = self._get_timestamp_col(dataset_type)
        id_col = "LCLid"

        before = len(combined)
        combined = combined.dropna(subset=[id_col, ts_col])
        metadata["invalid_records"] = before - len(combined)
        if metadata["invalid_records"]:
            logger.warning("Dropped %d rows with null keys", metadata["invalid_records"])

        dup_mask = combined.duplicated(subset=[id_col, ts_col], keep=False)
        n_dups = int(dup_mask.sum())
        metadata["duplicates_found"] = n_dups
        if n_dups:
            combined = combined.drop_duplicates(subset=[id_col, ts_col], keep="first")
            metadata["duplicates_removed"] = n_dups
            logger.warning("Removed %d duplicate rows (kept first)", n_dups)

        if sort:
            combined = combined.sort_values([id_col, ts_col]).reset_index(drop=True)
            logger.info("Sorted by (%s, %s)", id_col, ts_col)

        metadata["final_shape"] = combined.shape
        metadata["final_rows"] = len(combined)
        return combined, metadata

    def save_dataset(self, df: pd.DataFrame, name: str, out_dir: Path) -> Path:
        """Save a cleaned DataFrame to *out_dir*/<name>.parquet."""
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved %s  →  %s  (%.1f MB)", name, out_path, out_path.stat().st_size / 1e6)
        return out_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_timestamp_col(dataset_type: str) -> str:
        return "tstp" if dataset_type == "halfhourly_dataset" else "day"

    @staticmethod
    def _parse_timestamps(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        if dataset_type in ("daily_dataset", "hhblock_dataset"):
            df["day"] = pd.to_datetime(df["day"])
        elif dataset_type == "halfhourly_dataset":
            df["tstp"] = pd.to_datetime(df["tstp"])
            if "energy(kWh/hh)" in df.columns:
                df["energy(kWh/hh)"] = pd.to_numeric(
                    df["energy(kWh/hh)"].astype(str).str.strip(), errors="coerce"
                )
        return df

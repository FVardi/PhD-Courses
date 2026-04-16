"""
Task 01: Load and Validate Schema for London Smart Meters Dataset
=================================================================
Runnable EDA script.  Reusable logic lives in:
  - src/utils/data_loader.py   (DataLoader, SchemaValidator)
  - src/evaluation/plots.py    (plotting / diagnostic helpers)
  - src/utils/gaps.py          (gap quantification helpers)

Run from the Assignment 2 root:
    python scripts/task01_eda.py
=================================================================
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import DataLoader
from src.evaluation.plots import print_dataset_summary, plot_halfhourly_zoom
from src.utils.gaps import gaps_per_household, summarise_gaps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_ROOT = PROJECT_ROOT / "data" / "london_smart_meters"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

DATASETS = [
    ("halfhourly_dataset", [0, 1, 2], "tstp", "30min"),
    ("daily_dataset",      [0, 1, 2], "day",  "1D"),
    ("hhblock_dataset",    [0, 1, 2], "day",  "1D"),
]


def run_dataset(loader: DataLoader, dataset_type: str, blocks: list, ts_col: str, freq: str) -> dict:
    """Load, clean, summarise gaps, and save one dataset. Returns metadata."""
    logger.info("--- %s (blocks %s) ---", dataset_type, blocks)

    df, meta = loader.load_and_clean_dataset(dataset_type=dataset_type, blocks=blocks, sort=True)

    if df is None:
        logger.error("Failed to load %s", dataset_type)
        return meta

    logger.info(
        "Loaded: %d rows, %d households, %d blocks failed, "
        "%d duplicates removed, %d invalid records",
        meta["final_rows"],
        df["LCLid"].nunique(),
        len(meta["blocks_failed"]),
        meta["duplicates_removed"],
        meta["invalid_records"],
    )

    print_dataset_summary(df, dataset_type, meta)

    logger.info("Running gap analysis on %s…", dataset_type)
    gap_summary = summarise_gaps(df, ts_col=ts_col, freq=freq)
    logger.info(
        "Gap summary: %d gaps across %d/%d households, overall missingness %.2f%%",
        gap_summary["total_gaps"],
        gap_summary["n_households_with_gaps"],
        gap_summary["n_households"],
        gap_summary["overall_missingness_rate"] * 100,
    )

    gap_df = gaps_per_household(df, ts_col=ts_col, freq=freq)

    if dataset_type == "halfhourly_dataset":
        sample_ids = df["LCLid"].unique()[:5].tolist()
        logger.info("Plotting zoom levels for households: %s", sample_ids)
        fig = plot_halfhourly_zoom(df, household_ids=sample_ids)
        figures_dir = PREPROCESSED_DIR / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = figures_dir / "halfhourly_zoom.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        logger.info("Saved zoom plot → %s", fig_path)

    logger.info("Saving %s to %s…", dataset_type, PREPROCESSED_DIR)
    loader.save_dataset(df, dataset_type, PREPROCESSED_DIR)
    loader.save_dataset(gap_df, f"{dataset_type}_gaps_per_household", PREPROCESSED_DIR)

    return meta


def main() -> None:
    logger.info("=" * 70)
    logger.info("TASK 01: Load, Validate, and Summarise London Smart Meters")
    logger.info("=" * 70)

    loader = DataLoader(data_root=DATA_ROOT)
    all_meta = {}

    for dataset_type, blocks, ts_col, freq in DATASETS:
        all_meta[dataset_type] = run_dataset(loader, dataset_type, blocks, ts_col, freq)

    report = {"timestamp": datetime.now().isoformat(), **all_meta}
    report_path = PROJECT_ROOT / "scripts" / "validation_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Validation report saved to %s", report_path)

    logger.info("=" * 70)
    logger.info("TASK 01 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

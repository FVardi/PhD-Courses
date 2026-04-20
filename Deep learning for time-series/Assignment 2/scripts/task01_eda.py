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
# Reusable logic lives in:
#   src/utils/data_loader.py     (DataLoader, SchemaValidator)
#   src/evaluation/plots.py      (plotting / diagnostic helpers)
#   src/evaluation/gaps.py       (gap quantification helpers)

# %%
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import DataLoader
from src.evaluation.plots import (
    print_dataset_summary,
    plot_halfhourly_zoom,
    plot_acf_pacf,
    plot_daily_profile,
)
from src.evaluation.gaps import summarise_gaps


# %%

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_ROOT = PROJECT_ROOT / "data" / "london_smart_meters"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DATASETS = [
    ("halfhourly_dataset", [0, 1, 2], "tstp", "30min"),
    ("daily_dataset",      [0, 1, 2], "day",  "1D"),
    ("hhblock_dataset",    [0, 1, 2], "day",  "1D"),
]

# %%

def run_dataset(loader: DataLoader, dataset_type: str, blocks: list, ts_col: str, freq: str) -> dict:
    """Load, clean, summarise gaps, and save one dataset. Returns metadata."""
    logger.info("--- %s (blocks %s) ---", dataset_type, blocks)

    # --- Subtask 1: Load and validate schema ---
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

    # --- Subtask 2: Verify sampling regularity ---
    logger.info("Running gap analysis on %s…", dataset_type)
    gap_summary = summarise_gaps(df, ts_col=ts_col, freq=freq)
    logger.info(
        "Gap summary: %d gaps across %d/%d households, overall missingness %.2f%%",
        gap_summary["total_gaps"],
        gap_summary["n_households_with_gaps"],
        gap_summary["n_households"],
        gap_summary["overall_missingness_rate"] * 100,
    )

    # --- Subtask 3: Visualise household series ---
    if dataset_type == "halfhourly_dataset":
        sample_ids = df["LCLid"].unique()[:5].tolist()
        figures_dir = FIGURES_DIR
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Plotting zoom levels for households: %s", sample_ids)
        fig = plot_halfhourly_zoom(df, household_ids=sample_ids)
        fig.savefig(figures_dir / "halfhourly_zoom.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved zoom plot → %s", figures_dir / "halfhourly_zoom.png")

        logger.info("Plotting average daily profile…")
        fig_profile = plot_daily_profile(df)
        fig_profile.savefig(figures_dir / "daily_profile.png", dpi=150, bbox_inches="tight")
        plt.close(fig_profile)
        logger.info("Saved daily profile → %s", figures_dir / "daily_profile.png")

    # --- Subtask 4: ACF/PACF diagnostics ---
    if dataset_type == "halfhourly_dataset":
        figures_dir = FIGURES_DIR
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Plotting ACF/PACF (halfhourly) for households: %s", sample_ids[:3])
        fig_acf = plot_acf_pacf(df, household_ids=sample_ids[:3], lags=100,
                                timestamp_col="tstp", value_col="energy(kWh/hh)")
        fig_acf.savefig(figures_dir / "acf_pacf_halfhourly.png", dpi=150, bbox_inches="tight")
        plt.close(fig_acf)
        logger.info("Saved halfhourly ACF/PACF plot → %s", figures_dir / "acf_pacf_halfhourly.png")

    if dataset_type == "daily_dataset":
        sample_ids = df["LCLid"].unique()[:3].tolist()
        figures_dir = FIGURES_DIR
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Plotting ACF/PACF (daily) for households: %s", sample_ids)
        fig_acf_d = plot_acf_pacf(df, household_ids=sample_ids, lags=40,
                                   timestamp_col="day", value_col="energy_sum")
        fig_acf_d.savefig(figures_dir / "acf_pacf_daily.png", dpi=150, bbox_inches="tight")
        plt.close(fig_acf_d)
        logger.info("Saved daily ACF/PACF plot → %s", figures_dir / "acf_pacf_daily.png")

    # --- Subtask 5: Dataset summary ---
    print_dataset_summary(df, dataset_type, meta)

    return meta
# %%

def main() -> None:
    logger.info("=" * 70)
    logger.info("TASK 01: Load, Validate, and Summarise London Smart Meters")
    logger.info("=" * 70)

    loader = DataLoader(data_root=DATA_ROOT)
    all_meta = {}

    for dataset_type, blocks, ts_col, freq in DATASETS:
        all_meta[dataset_type] = run_dataset(loader, dataset_type, blocks, ts_col, freq)

    report = {"timestamp": datetime.now().isoformat(), **all_meta}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "task01_validation_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Validation report saved to %s", report_path)

    logger.info("=" * 70)
    logger.info("TASK 01 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

# %%

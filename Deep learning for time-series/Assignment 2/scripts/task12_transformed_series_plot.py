"""
Task 12: Raw vs transformed target series.

For the five example households used throughout the diagnostic tasks,
plots energy_imputed_seasonal (raw) alongside the Log + Deseasonalise
transform that is applied to the target in all models (Tasks 6-10).

Transform is fitted on each household's own training data only.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import VALUE_COL, TRAIN_END, TEST_START, save_fig
from src.transforms.transforms import ComposedTransform, DeseasonalisingTransform, LogTransform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "report" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

LCLIDS = ["MAC000002", "MAC000003", "MAC000004", "MAC000005", "MAC000006"]

logger.info("Loading data for %d example households …", len(LCLIDS))
data = pd.read_parquet(FEATURES_DIR, filters=[("LCLid", "in", LCLIDS)], columns=[VALUE_COL])

fig, axes = plt.subplots(len(LCLIDS), 2, figsize=(16, 3 * len(LCLIDS)), sharex=False)

for i, lclid in enumerate(LCLIDS):
    hh     = data.xs(lclid, level="LCLid")[VALUE_COL].sort_index()
    train_s = hh.loc[hh.index < TRAIN_END]

    transform = ComposedTransform([LogTransform(), DeseasonalisingTransform(period=48)])
    transform.fit(train_s)
    transformed = transform.transform(hh)

    for ax, series, title, color, ylabel in [
        (axes[i, 0], hh,          f"{lclid} — Raw",                "steelblue", "kWh/hh"),
        (axes[i, 1], transformed, f"{lclid} — Log + Deseasonalise", "seagreen",  "Transformed"),
    ]:
        ax.plot(series.index, series.values, linewidth=0.6, color=color)
        ax.axvline(TRAIN_END,  color="darkorange", linewidth=1, linestyle="--",
                   label="Train/Val" if i == 0 else None)
        ax.axvline(TEST_START, color="purple",     linewidth=1, linestyle="--",
                   label="Test start" if i == 0 else None)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=20)

    axes[i, 1].axhline(0, color="k", linewidth=0.4, linestyle=":")

axes[0, 0].legend(fontsize=7, ncol=2)

fig.suptitle(
    "Raw vs Log + Deseasonalise transform (fitted on training data only)",
    fontsize=12, y=1.002,
)
plt.tight_layout()
save_fig(fig, "task12_raw_vs_transformed.png", ARTIFACTS_DIR)
logger.info("Task 12 complete.")

"""
Task 15: LightGBM without target transform vs Log + Deseasonalise.

Refits the tuned global LightGBM (Task 10 best params, Task 09 best
encoding) without the Log + Deseasonalise target transformation, and
compares per-household test MASE against the transformed version.

LightGBM is tree-based and does not require variance stabilisation,
so it can be applied directly to the raw kWh target.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _experiment_setup import TARGET_TRANSFORM, load_tuned_setup, build_wrapper
from src.pipeline import per_hh_metrics, save_fig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "report" / "artifacts"
ACORN_PATH    = PROJECT_ROOT / "data" / "london_smart_meters" / "informations_households.csv"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Load shared setup -------------------------------------------------------

logger.info("Setting up cohort and encoding …")
cohort_ids, tr, va, te, fc, mc, best10_params = load_tuned_setup(
    ARTIFACTS_DIR, FEATURES_DIR, ACORN_PATH
)

# --- Fit with transform (Log + Deseasonalise) ---------------------------------

logger.info("Fitting with Log + Deseasonalise transform …")
w_transform = build_wrapper(best10_params, fc, mc, target_transformer=TARGET_TRANSFORM)
w_transform.fit(tr)
m_transform = per_hh_metrics(
    w_transform.predict(va), w_transform.predict(te), va, te, tr, cohort_ids
)
logger.info(
    "With transform  — Test MASE: mean=%.4f  median=%.4f  std=%.4f",
    m_transform["test_mase"].mean(),
    m_transform["test_mase"].median(),
    m_transform["test_mase"].std(),
)

# --- Fit without transform (raw kWh target) -----------------------------------

logger.info("Fitting without target transform …")
w_raw = build_wrapper(best10_params, fc, mc, target_transformer=None)
w_raw.fit(tr)
m_raw = per_hh_metrics(
    w_raw.predict(va), w_raw.predict(te), va, te, tr, cohort_ids
)
logger.info(
    "Without transform — Test MASE: mean=%.4f  median=%.4f  std=%.4f",
    m_raw["test_mase"].mean(),
    m_raw["test_mase"].median(),
    m_raw["test_mase"].std(),
)

# --- Violin comparison -------------------------------------------------------

vals_transform = m_transform["test_mase"].dropna()
vals_raw       = m_raw["test_mase"].dropna()

fig, ax = plt.subplots(figsize=(7, 5))

for pos, vals, color, label in [
    (0, vals_transform, "steelblue", "With transform\n(Log + Deseasonalise)"),
    (1, vals_raw,       "tomato",    "No transform\n(raw kWh target)"),
]:
    parts = ax.violinplot(vals.values, positions=[pos], showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    ax.scatter(np.full(len(vals), pos), vals.values, s=10, color=color, alpha=0.4, zorder=3)

ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1, label="MASE = 1 (naïve)")
ax.set_xticks([0, 1])
ax.set_xticklabels(["With transform\n(Log + Deseasonalise)", "No transform\n(raw kWh target)"])
ax.set_ylabel("Test MASE")
ax.set_title(
    f"Tuned LightGBM — transform vs no transform  (n={len(cohort_ids)} households)"
)
ax.legend()
fig.tight_layout()
save_fig(fig, "task15_no_transform_comparison.png", ARTIFACTS_DIR)

# --- Summary -----------------------------------------------------------------

gain = vals_transform.median() - vals_raw.median()
direction = "better" if gain > 0 else "worse"
logger.info(
    "Transform %s by %.4f MASE points (median).  "
    "With=%.4f  Without=%.4f",
    direction, abs(gain), vals_transform.median(), vals_raw.median(),
)

logger.info("Task 15 complete.")

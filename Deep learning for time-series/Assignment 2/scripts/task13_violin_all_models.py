"""
Task 13: Combined violin plot — per-household test MASE for all LightGBM models.

Models compared:
  (1) Local LightGBM   (Task 07) — per-household, best config from Task 06
  (2) Global LightGBM  (Task 08) — pooled, no meta-features
  (3) Global + meta    (Task 09) — best encoding variant
  (4) Tuned global     (Task 10) — best hyperparameters from Optuna

Tasks 07 and 08 per-household metrics are loaded from saved CSVs.
Tasks 09 and 10 models are refitted here to collect per-household MASE.
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
from src.pipeline import LGBM_DEFAULTS, per_hh_metrics, save_fig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "report" / "artifacts"
ACORN_PATH    = PROJECT_ROOT / "data" / "london_smart_meters" / "informations_households.csv"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Load pre-saved metrics for tasks 07 and 08 ------------------------------

logger.info("Loading pre-saved per-household metrics (tasks 07 and 08) …")
t07 = pd.read_csv(ARTIFACTS_DIR / "task07_per_household_metrics.csv", index_col="LCLid")
t08 = pd.read_csv(ARTIFACTS_DIR / "task08_per_household_metrics.csv", index_col="LCLid")

# --- Load shared setup -------------------------------------------------------

logger.info("Setting up cohort and encoding …")
cohort_ids, tr, va, te, fc, mc, best10_params = load_tuned_setup(
    ARTIFACTS_DIR, FEATURES_DIR, ACORN_PATH
)

# --- Fit task09 best variant (LGBM defaults, best encoding) ------------------

logger.info("Fitting task09 model (LGBM defaults + best encoding) …")
w09 = build_wrapper(LGBM_DEFAULTS, fc, mc)
w09.fit(tr)
t09 = per_hh_metrics(w09.predict(va), w09.predict(te), va, te, tr, cohort_ids)
logger.info("Task09 — Test MASE: mean=%.4f  median=%.4f", t09["test_mase"].mean(), t09["test_mase"].median())

# --- Fit task10 tuned model --------------------------------------------------

logger.info("Fitting task10 model (tuned hyperparameters) …")
w10 = build_wrapper(best10_params, fc, mc)
w10.fit(tr)
t10 = per_hh_metrics(w10.predict(va), w10.predict(te), va, te, tr, cohort_ids)
logger.info("Task10 — Test MASE: mean=%.4f  median=%.4f", t10["test_mase"].mean(), t10["test_mase"].median())

# --- Combined violin plot ----------------------------------------------------

models = {
    "Local\n(Task 07)":        t07["test_mase"].dropna(),
    "Global\n(Task 08)":       t08["test_mase"].dropna(),
    "Global+meta\n(Task 09)":  t09["test_mase"].dropna(),
    "Tuned global\n(Task 10)": t10["test_mase"].dropna(),
}
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
positions = list(range(len(models)))

fig, ax = plt.subplots(figsize=(10, 5))

for pos, (label, vals), color in zip(positions, models.items(), colors):
    parts = ax.violinplot(vals.values, positions=[pos], showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    ax.scatter(
        np.full(len(vals), pos), vals.values,
        s=8, color=color, alpha=0.4, zorder=3,
    )

ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1, label="MASE = 1 (naïve)")
ax.set_xticks(positions)
ax.set_xticklabels(list(models.keys()))
ax.set_ylabel("Test MASE")
ax.set_title(f"Per-household Test MASE — all LightGBM models  (n={len(cohort_ids)} households)")
ax.legend()
fig.tight_layout()
save_fig(fig, "task13_violin_all_models.png", ARTIFACTS_DIR)

# --- Print summary table -----------------------------------------------------

summary = pd.DataFrame({
    label.replace("\n", " "): {
        "mean":   round(vals.mean(),   4),
        "median": round(vals.median(), 4),
        "std":    round(vals.std(),    4),
        "pct < 1 (beat naïve)": f"{(vals < 1.0).mean() * 100:.1f}%",
    }
    for label, vals in models.items()
}).T
logger.info("Summary:\n%s", summary.to_string())

logger.info("Task 13 complete.")

"""
Task 14: Recursive multi-step forecasting.

Uses the tuned global LightGBM (Task 10) to generate H-step recursive
forecasts from the start of the test period.

At each step i, lag_k features are replaced with predictions from step i-k
whenever i-k >= 0.  Rolling, EWMA, and calendar features are held at their
pre-computed values (from the features parquet).

Plots true values, standard 1-step-ahead predictions, and the recursive
H-step forecast for N_PLOT households.
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

from _experiment_setup import load_tuned_setup, build_wrapper
from src.pipeline import VALUE_COL, save_fig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "report" / "artifacts"
ACORN_PATH    = PROJECT_ROOT / "data" / "london_smart_meters" / "informations_households.csv"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

H      = 96   # 96 × 30 min = 48 hours
N_PLOT = 3
LAG_KEYS = [1, 2, 3, 4, 5, 6, 48, 336]

# --- Load setup and fit model ------------------------------------------------

logger.info("Setting up cohort and encoding …")
cohort_ids, tr, va, te, fc, mc, best10_params = load_tuned_setup(
    ARTIFACTS_DIR, FEATURES_DIR, ACORN_PATH
)

logger.info("Fitting tuned global model …")
wrapper = build_wrapper(best10_params, fc, mc)
wrapper.fit(tr)
logger.info("Model fitted.")

# --- Recursive prediction function -------------------------------------------

def recursive_predict(wrapper, hh_test: pd.DataFrame, H: int) -> pd.Series:
    """Recursively predict H steps for a single household.

    Uses actual lag values for steps where no prediction is available yet
    (i.e. i - k < 0).  All other features are unchanged from the pre-computed
    test feature table.
    """
    df = hh_test.sort_index(level="tstp").head(H).copy()
    pred_history: dict[int, float] = {}

    for i in range(len(df)):
        row = df.iloc[[i]].copy()

        for k in LAG_KEYS:
            j = i - k
            if j >= 0 and j in pred_history:
                col = f"lag_{k}"
                if col in row.columns:
                    row[col] = pred_history[j]

        result = wrapper.predict(row)
        pred_history[i] = float(result.iloc[0]) if len(result) > 0 else float("nan")

    return pd.Series(list(pred_history.values()), index=df.index)

# --- Plot for N_PLOT households -----------------------------------------------

plot_ids = cohort_ids[:N_PLOT]

for lclid in plot_ids:
    logger.info("Generating %d-step recursive forecast for %s …", H, lclid)

    mask     = te.index.get_level_values("LCLid") == lclid
    hh_test  = te.loc[mask]

    y_actual   = hh_test[VALUE_COL].sort_index().iloc[:H]
    y_onestep  = wrapper.predict(hh_test).reindex(y_actual.index)
    y_recursive = recursive_predict(wrapper, hh_test, H=H).reindex(y_actual.index)

    t = y_actual.index.get_level_values("tstp")

    # Metrics
    err_1   = (y_actual.values - y_onestep.values)
    err_rec = (y_actual.values - y_recursive.values)
    mae_1   = float(np.nanmean(np.abs(err_1)))
    mae_rec = float(np.nanmean(np.abs(err_rec)))
    logger.info(
        "%s — 1-step MAE=%.4f  recursive MAE=%.4f  (ratio=%.2f×)",
        lclid, mae_1, mae_rec, mae_rec / mae_1 if mae_1 > 0 else float("nan"),
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, y_actual.values,    color="black",     linewidth=1.2,  label="Actual")
    ax.plot(t, y_onestep.values,   color="steelblue", linewidth=1.0,  linestyle="--",
            label=f"1-step-ahead  (MAE={mae_1:.3f})")
    ax.plot(t, y_recursive.values, color="tomato",    linewidth=1.0,
            label=f"{H}-step recursive  (MAE={mae_rec:.3f})")

    ax.set_title(
        f"{lclid} — 1-step-ahead vs {H}-step recursive forecast ({H * 0.5:.0f} h horizon)",
        fontsize=10,
    )
    ax.set_ylabel("Energy (kWh/hh)")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    save_fig(fig, f"task14_recursive_{lclid}.png", ARTIFACTS_DIR)

logger.info("Task 14 complete. Plots saved to %s", ARTIFACTS_DIR)

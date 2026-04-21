"""
Task 08: Global Model (Pooled Multi-Household LightGBM)

Pipeline:
  1. Pool cohort households into one train / val / test DataFrame.
  2. Encode household_id as pd.Categorical (native LightGBM feature).
  3. Train one global LightGBM model on the pooled training set.
  4. Evaluate globally (pooled) and per-household on val and test.
  5. Compare against Task 07 local cohort results.
  6. Scatter plot + summary table + trade-off interpretation.

Validation checks:
  (a) Splits applied before pooling — no data leakage across the boundary.
  (b) Features computed within-household only (features parquet).
  (c) No future data seen during training.
  (d) MASE denominator computed per household from training data only.
"""

# %%
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    VALUE_COL,
    LGBM_DEFAULTS,
    make_feature_config, make_missing_config, make_lgbm_model_config,
    per_hh_metrics,
    load_cohort, load_splits, add_lclid_enc,
    save_fig, save_csv,
)
from src.forecasting import MLForecast
from src.transforms.transforms import DeseasonalisingTransform

# %%

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "results" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Run mode ──────────────────────────────────────────────────────────────────
# QUICK_RUN=True  → 5 households; fast for inspection.
# QUICK_RUN=False → 50 households; recommended before submitting.
QUICK_RUN      = True
MAX_HOUSEHOLDS = 5 if QUICK_RUN else 50

# %%
# --- Step 1: Cohort + data ---------------------------------------------------

cohort_ids = load_cohort(ARTIFACTS_DIR, MAX_HOUSEHOLDS)
train_pool, val_pool, test_pool = load_splits(FEATURES_DIR, cohort_ids)

# %%
# --- Step 2: Add lclid_enc as native categorical feature ---------------------
#
# LCLid is mapped to an integer and cast to pd.Categorical so LightGBM treats
# it as a nominal variable (not ordinal numeric).

_lclid_map = add_lclid_enc(train_pool, val_pool, test_pool, cohort_ids)
logger.info("lclid_enc: %d households, range [0, %d]", len(_lclid_map), len(_lclid_map) - 1)

# %%
# --- Step 3: Configs ---------------------------------------------------------
#
# LightGBM baseline hyperparameters — deliberately conservative.
# Tuning is deferred to Task 10.  Documented defaults:
#   num_leaves=31       : LightGBM default; balances expressiveness vs overfitting
#   n_estimators=500    : enough iterations to converge at lr=0.05
#   learning_rate=0.05  : standard moderate step size
#   min_child_samples=20: prevents tiny leaves on noisy households
#   subsample=0.8       : row bagging per tree for regularisation
#   colsample_bytree=0.8: feature bagging per tree for regularisation

logger.info("LightGBM hyperparameters: %s", LGBM_DEFAULTS)

feature_config = make_feature_config(native_categorical=True)
missing_config = make_missing_config()
model_config   = make_lgbm_model_config()

# %%
# --- Step 4: Fit global model ------------------------------------------------

wrapper = MLForecast(
    model_config         = model_config,
    feature_config       = feature_config,
    missing_value_config = missing_config,
    target_transformer   = DeseasonalisingTransform(period=48),
)
logger.info("Fitting global model on pooled training set (%d rows) …", len(train_pool))
wrapper.fit(train_pool)
logger.info("Global model fitted.")

# %%
# --- Step 5: Pooled predictions + metrics ------------------------------------

y_hat_val  = wrapper.predict(val_pool)
y_hat_test = wrapper.predict(test_pool)

y_val_true  = val_pool[VALUE_COL].reindex(y_hat_val.index)
y_test_true = test_pool[VALUE_COL].reindex(y_hat_test.index)

pooled_val_mae   = (y_val_true  - y_hat_val).abs().mean()
pooled_val_rmse  = np.sqrt(((y_val_true  - y_hat_val) ** 2).mean())
pooled_test_mae  = (y_test_true - y_hat_test).abs().mean()
pooled_test_rmse = np.sqrt(((y_test_true - y_hat_test) ** 2).mean())

logger.info(
    "Pooled — Val  MAE=%.4f  RMSE=%.4f | Test MAE=%.4f  RMSE=%.4f",
    pooled_val_mae, pooled_val_rmse, pooled_test_mae, pooled_test_rmse,
)

# %%
# --- Step 6: Per-household metrics (MASE denom from training only) -----------

global_metrics = per_hh_metrics(
    y_hat_val, y_hat_test, val_pool, test_pool, train_pool, cohort_ids,
)
save_csv(global_metrics, "task08_per_household_metrics.csv", ARTIFACTS_DIR)

logger.info(
    "Per-household Test MASE — mean=%.4f  median=%.4f  std=%.4f",
    global_metrics["test_mase"].mean(),
    global_metrics["test_mase"].median(),
    global_metrics["test_mase"].std(),
)

# %%
# --- Step 7: Load Task 07 local results for comparison -----------------------

_local_path = ARTIFACTS_DIR / "task07_per_household_metrics.csv"
local_metrics: pd.DataFrame | None = None
if _local_path.exists():
    local_metrics = pd.read_csv(_local_path, index_col="LCLid")
    logger.info("Loaded Task 07 local metrics: %d households", len(local_metrics))
else:
    logger.warning("Task 07 metrics not found — comparison skipped.")

# %%
# --- Step 8: Summary comparison table ----------------------------------------

if local_metrics is not None and "test_mase" in local_metrics.columns:
    shared = global_metrics.index.intersection(local_metrics.index)
    loc = local_metrics.loc[shared]
    glo = global_metrics.loc[shared]

    n_global_better = (glo["test_mase"] < loc["test_mase"]).sum()
    n_local_better  = (loc["test_mase"] < glo["test_mase"]).sum()

    summary = pd.DataFrame({
        "metric": ["val_mae", "val_rmse", "test_mae", "test_rmse",
                   "test_mase_mean", "test_mase_median"],
        "local (task07)": [
            loc["val_mae"].mean()  if "val_mae"  in loc.columns else np.nan,
            loc["val_rmse"].mean() if "val_rmse" in loc.columns else np.nan,
            loc["test_mae"].mean(), loc["test_rmse"].mean(),
            loc["test_mase"].mean(), loc["test_mase"].median(),
        ],
        "global (task08)": [
            glo["val_mae"].mean(), glo["val_rmse"].mean(),
            glo["test_mae"].mean(), glo["test_rmse"].mean(),
            glo["test_mase"].mean(), glo["test_mase"].median(),
        ],
    }).set_index("metric").round(4)

    save_csv(summary, "task08_comparison_summary.csv", ARTIFACTS_DIR)
    logger.info(
        "Comparison (%d shared households): global better=%d  local better=%d",
        len(shared), n_global_better, n_local_better,
    )
    logger.info("Summary:\n%s", summary.to_string())

# %%
# --- Step 9: Scatter plot — local vs global Test MASE ------------------------

if local_metrics is not None and "test_mase" in local_metrics.columns:
    shared = global_metrics.index.intersection(local_metrics.index)
    x = local_metrics.loc[shared, "test_mase"]
    y = global_metrics.loc[shared, "test_mase"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.6, edgecolors="k", linewidths=0.4, s=50)
    lim = (min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.05)
    ax.plot(lim, lim, "r--", lw=1, label="parity")
    ax.set_xlabel("Local Test MASE (Task 07)"); ax.set_ylabel("Global Test MASE (Task 08)")
    ax.set_title("Local vs Global Model: Test MASE per Household")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.legend()
    fig.tight_layout()
    save_fig(fig, "task08_local_vs_global_mase.png", ARTIFACTS_DIR)

# %%
# --- Step 10: Per-household MASE distribution --------------------------------

fig, ax = plt.subplots(figsize=(6, 4))
ax.violinplot([global_metrics["test_mase"].dropna().values], positions=[1], showmedians=True)
ax.set_xticks([1]); ax.set_xticklabels(["Global LightGBM"])
ax.set_ylabel("Test MASE")
ax.set_title("Global Model — per-household Test MASE")
fig.tight_layout()
save_fig(fig, "task08_global_mase_distribution.png", ARTIFACTS_DIR)

# %%
# --- Step 11: Trade-off interpretation ---------------------------------------

logger.info(
    """
Trade-offs — global vs local models
====================================
Advantages of the global model:
  + Trains on ~%d× the data of any single local model → better gradient estimates.
  + Shared temporal patterns (time-of-day, week) learned from many households.
  + One model to deploy; no per-household retraining.

Disadvantages:
  - Household heterogeneity: a single model may underfit outlier households.
  - lclid_enc cannot generalise to unseen households (cold-start).
  - Hyperparameters tuned globally; individual tuning may help hard cases (Task 10).
  - Inference cost scales with pooled feature space.
""",
    len(cohort_ids),
)

logger.info("Task 08 complete. Artifacts saved to %s", ARTIFACTS_DIR)

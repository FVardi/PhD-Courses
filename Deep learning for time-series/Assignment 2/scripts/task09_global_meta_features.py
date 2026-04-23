"""
Task 09: Global Model with Household Meta-Features

Enrich the pooled dataset with household-level meta-features and compare
four categorical encoding strategies.  The goal is to test whether static
household attributes improve the global model beyond using household_id alone.

Variants (same LightGBM defaults throughout — encoding effect isolated):
  (a) Baseline    — no meta-features, identical to Task 8
  (b) OHE         — one-hot encode meta-features via the wrapper
  (c) Count       — replace each category with its training-set frequency
  (d) Target enc  — replace each category with its mean training target

Validation checks:
  (a) Meta-features joined by LCLid (household_id).
  (b) Same train / val / test split as Task 8.
  (c) Target encoding fitted on training targets only.
  (d) Unseen categories: global training mean (target enc), 0 (count enc),
      "ignore" bin (OHE via handle_unknown="ignore").
"""

# %%
import json
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
    VALUE_COL, LGBM_DEFAULTS,
    make_feature_config, make_missing_config, make_lgbm_model_config,
    per_hh_metrics,
    load_cohort, load_splits, add_lclid_enc,
    save_fig, save_csv,
)
from src.forecasting import MLForecast
from src.transforms.transforms import ComposedTransform, DeseasonalisingTransform, LogTransform

# %%

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "report" / "artifacts"
ACORN_PATH    = PROJECT_ROOT / "data" / "london_smart_meters" / "informations_households.csv"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Run mode ──────────────────────────────────────────────────────────────────
# QUICK_RUN=True  → 5 households; fast for inspection.
# QUICK_RUN=False → 50 households; recommended before submitting.
QUICK_RUN      = False
MAX_HOUSEHOLDS = 5 if QUICK_RUN else 50

META_COLS = ["stdorToU", "Acorn", "Acorn_grouped"]

# %%
# --- Step 1: Cohort + data ---------------------------------------------------

cohort_ids = load_cohort(ARTIFACTS_DIR, MAX_HOUSEHOLDS)
train_base, val_base, test_base = load_splits(FEATURES_DIR, cohort_ids)
_lclid_map = add_lclid_enc(train_base, val_base, test_base, cohort_ids)

# %%
# --- Step 2: Load household meta-features ------------------------------------
#
# stdorToU, Acorn, Acorn_grouped from informations_households.csv.
# Missing / unseen values are filled with "Unknown" as a documented fallback.

if not ACORN_PATH.exists():
    raise FileNotFoundError(f"Household metadata not found at {ACORN_PATH}.")

meta = (
    pd.read_csv(ACORN_PATH, usecols=["LCLid"] + META_COLS)
    .set_index("LCLid")
    .loc[lambda df: ~df.index.duplicated(keep="first")]
    .fillna("Unknown")
    .reindex(cohort_ids)
    .fillna("Unknown")
)
n_unknown = (meta == "Unknown").any(axis=1).sum()
logger.info(
    "Meta-features: %d households, %d with ≥1 unknown value  (cols: %s)",
    len(meta), n_unknown, META_COLS,
)


def _join_meta(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    lclids = pool.index.get_level_values("LCLid")
    for col in META_COLS:
        pool[col] = lclids.map(meta[col]).values
    return pool


# %%
# --- Step 3: Shared fitting helper -------------------------------------------

def _fit_and_eval(
    variant_name: str,
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    feature_config,
) -> dict:
    logger.info("Fitting variant: %s …", variant_name)
    wrapper = MLForecast(
        model_config         = make_lgbm_model_config(),
        feature_config       = feature_config,
        missing_value_config = make_missing_config(),
        target_transformer   = ComposedTransform([LogTransform(), DeseasonalisingTransform(period=48)]),
    )
    wrapper.fit(train)
    yv_hat = wrapper.predict(val)
    yt_hat = wrapper.predict(test)

    hh_df = per_hh_metrics(yv_hat, yt_hat, val, test, train, cohort_ids)
    vm = hh_df[["val_mae", "val_rmse", "val_mase"]].mean()
    tm = hh_df[["test_mae", "test_rmse", "test_mase"]].mean()

    result = {
        "variant":   variant_name,
        "val_mae":   round(vm["val_mae"],   4),
        "val_rmse":  round(vm["val_rmse"],  4),
        "val_mase":  round(vm["val_mase"],  4),
        "test_mae":  round(tm["test_mae"],  4),
        "test_rmse": round(tm["test_rmse"], 4),
        "test_mase": round(tm["test_mase"], 4),
    }
    logger.info(
        "%-20s  Val MAE=%.4f  MASE=%.4f | Test MAE=%.4f  MASE=%.4f",
        variant_name, result["val_mae"], result["val_mase"],
        result["test_mae"], result["test_mase"],
    )
    return result


# %%
# --- Step 4a: Variant (a) Baseline — no meta-features (= Task 8) -------------

results = []
results.append(_fit_and_eval(
    "a_baseline",
    train_base, val_base, test_base,
    make_feature_config(native_categorical=True),
))

# %%
# --- Step 4b: Variant (b) OHE encoding ---------------------------------------
#
# Meta-features added as categorical_features → OHE by the wrapper.
# handle_unknown="ignore" silently zeros out unseen categories.

train_ohe = _join_meta(train_base)
val_ohe   = _join_meta(val_base)
test_ohe  = _join_meta(test_base)

results.append(_fit_and_eval(
    "b_ohe",
    train_ohe, val_ohe, test_ohe,
    make_feature_config(native_categorical=True, extra_categorical=META_COLS),
))

# %%
# --- Step 4c: Variant (c) Count encoding -------------------------------------
#
# Replace each category with its row count in the training set.
# Unseen categories → 0 (documented fallback).
# Encoding statistics from training split only.

def _count_encode(tr, va, te, cols):
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for col in cols:
        counts  = tr[col].value_counts()
        enc_col = f"{col}_cnt"
        for pool in (tr, va, te):
            pool[enc_col] = pool[col].map(counts).fillna(0).astype(float)
            pool.drop(columns=[col], inplace=True)
    return tr, va, te


train_cnt, val_cnt, test_cnt = _count_encode(
    _join_meta(train_base), _join_meta(val_base), _join_meta(test_base), META_COLS,
)
cnt_cols = [f"{c}_cnt" for c in META_COLS]

results.append(_fit_and_eval(
    "c_count",
    train_cnt, val_cnt, test_cnt,
    make_feature_config(native_categorical=True, extra_continuous=cnt_cols),
))

# %%
# --- Step 4d: Variant (d) Target encoding ------------------------------------
#
# Replace each category with the mean training target for that group.
# Unseen categories → global training mean (documented fallback).
# Encoding statistics from training split ONLY (validation check c).

def _target_encode(tr, va, te, cols, target_col):
    tr, va, te = tr.copy(), va.copy(), te.copy()
    global_mean = tr[target_col].mean()
    for col in cols:
        means   = tr.groupby(col)[target_col].mean()
        enc_col = f"{col}_te"
        for pool in (tr, va, te):
            pool[enc_col] = pool[col].map(means).fillna(global_mean).astype(float)
            pool.drop(columns=[col], inplace=True)
    return tr, va, te


train_te, val_te, test_te = _target_encode(
    _join_meta(train_base), _join_meta(val_base), _join_meta(test_base),
    META_COLS, VALUE_COL,
)
te_cols = [f"{c}_te" for c in META_COLS]

results.append(_fit_and_eval(
    "d_target_enc",
    train_te, val_te, test_te,
    make_feature_config(native_categorical=True, extra_continuous=te_cols),
))

# %%
# --- Step 5: Comparison table ------------------------------------------------

comparison = pd.DataFrame(results).set_index("variant")
save_csv(comparison, "task09_encoding_comparison.csv", ARTIFACTS_DIR)
logger.info("Comparison table:\n%s", comparison.to_string())

best_variant = comparison["val_mase"].idxmin()
logger.info("Best variant by Val MASE: %s", best_variant)

# %%
# --- Step 6: Bar chart -------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric, title in zip(axes, ["val_mase", "test_mase"], ["Validation MASE", "Test MASE"]):
    vals = comparison[metric]
    bars = ax.bar(vals.index, vals.values, color="steelblue", edgecolor="k", linewidth=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1, label="MASE = 1 (naïve)")
    ax.set_title(title); ax.set_ylabel("MASE"); ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
fig.suptitle("Task 09 — Encoding Variant Comparison", fontsize=12)
fig.tight_layout()
save_fig(fig, "task09_encoding_comparison.png", ARTIFACTS_DIR)

# %%
# --- Step 7: Save best variant info for Task 10 ------------------------------

best_row  = comparison.loc[best_variant]
best_info = {
    "variant":       best_variant,
    "val_mae":       float(best_row["val_mae"]),
    "val_mase":      float(best_row["val_mase"]),
    "test_mae":      float(best_row["test_mae"]),
    "test_mase":     float(best_row["test_mase"]),
    "meta_cols":     META_COLS,
    "lgbm_defaults": LGBM_DEFAULTS,
}
_best_path = ARTIFACTS_DIR / "task09_best_variant.json"
with open(_best_path, "w") as f:
    json.dump(best_info, f, indent=2)
logger.info("Best variant config saved → %s", _best_path)

# %%
# --- Step 8: Interpretation --------------------------------------------------

logger.info(
    """
Interpretation
==============
Meta-features: %s

(a) Baseline   — no meta-features; only household_id as a categorical feature. No generalization across households; unseen households in val/test → "Unknown" bin. Worst in all cases.

(b) OHE        — generalizes across households of the same category but applied no weights. Seems to be the most robust and most accurate variant, improving over baseline without overfitting.

(c) Count enc  — generalizes across households of the same category, weighted by training frequency. Similar to OHE but slightly worse.

(d) Target enc — generalizes across households of the same category, weighted by training target mean. Almost as good as OHE and count in everything but MASE.

OHE encoding is deemed best for its low validation MASE and test MASE as well as its generalization strength. Since there is not a lot of variance between groups (Kruskall-Wallis test) there is not a big difference in the encoding of the groups.

The chosen meta features procide useful information beyond household_id alone.

Best variant: %s  (Val MASE=%.4f)
""",
    META_COLS, best_variant, float(best_row["val_mase"]),
)

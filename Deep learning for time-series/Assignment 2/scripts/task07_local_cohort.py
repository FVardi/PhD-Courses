"""
Task 07: Local Cohort Forecasting Study

Pipeline:
  1. Select a cohort from the Task 4 quality screen.
  2. Train one local model per household (best config from Task 6).
  3. Collect per-household val/test MAE, RMSE, MASE.
  4. Summarise distribution and plot violin.
  5. Investigate hard households (top-10% Test MASE).
  6. Acorn subgroup comparison (Kruskal-Wallis) if metadata is available.

Parallelised with joblib — each household is fitted independently.
Best model config is loaded from results/artifacts/task06_best_model.json.
"""

# %%
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scs
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    VALUE_COL, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    make_feature_config, make_missing_config,
    mase_denom, eval_metrics,
    load_cohort,
    save_fig, save_csv,
)
from src.configs import ModelConfig
from src.forecasting import MLForecast
from src.transforms.transforms import DeseasonalisingTransform

# %%

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
IMPUTED_PATH  = PROJECT_ROOT / "data" / "preprocessed" / "halfhourly_imputed.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "results" / "artifacts"
ACORN_PATH    = PROJECT_ROOT / "data" / "london_smart_meters" / "informations_households.csv"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Run mode ──────────────────────────────────────────────────────────────────
# QUICK_RUN=True  → 5 households; fast for inspection.
# QUICK_RUN=False → 50 households; recommended before submitting.
QUICK_RUN      = False
MAX_HOUSEHOLDS = 5  if QUICK_RUN else 50
N_JOBS         = -1

# %%
# --- Step 1: Load best model config from Task 6 ------------------------------

_cfg_path = ARTIFACTS_DIR / "task06_best_model.json"
if not _cfg_path.exists():
    raise FileNotFoundError(f"Best model config not found at {_cfg_path}. Run task06 first.")

with open(_cfg_path) as _f:
    BEST_CFG = json.load(_f)

logger.info(
    "Loaded best model config: family=%s  params=%s  val_mae=%.4f",
    BEST_CFG["family"], BEST_CFG["params"], BEST_CFG["val_mae"],
)

# %%
# --- Step 2: Cohort selection ------------------------------------------------

cohort_ids = load_cohort(ARTIFACTS_DIR, MAX_HOUSEHOLDS)
logger.info("Cohort: %d households", len(cohort_ids))

# %%
# --- Step 3: Acorn metadata --------------------------------------------------

acorn_df: pd.DataFrame | None = None
if ACORN_PATH.exists():
    _acorn_raw = pd.read_csv(ACORN_PATH)
    if "LCLid" in _acorn_raw.columns and "Acorn_grouped" in _acorn_raw.columns:
        acorn_df = _acorn_raw.set_index("LCLid")[["Acorn_grouped"]].copy()
    else:
        logger.warning("informations_households.csv missing LCLid/Acorn_grouped.")
else:
    logger.info("Acorn metadata not found — subgroup analysis skipped.")

# %%
# --- Shared configs (pickleable — passed to worker) --------------------------

FEATURE_CONFIG = make_feature_config()
MISSING_CONFIG = make_missing_config()


# %%
# --- Worker function (module-level for joblib pickling) ----------------------

def _build_estimator(family: str, params: dict):
    _map = {
        "LinearRegression": LinearRegression,
        "Ridge":            Ridge,
        "Lasso":            Lasso,
        "DecisionTree":     DecisionTreeRegressor,
        "RandomForest":     RandomForestRegressor,
        "LightGBM":         LGBMRegressor,
    }
    if family not in _map:
        raise ValueError(f"Unknown model family: {family}")
    return _map[family](**params)


def _fit_one_household(
    lclid: str,
    features_dir_str: str,
    feature_config: "FeatureConfig",
    missing_config: "MissingValueConfig",
    model_family: str,
    model_params: dict,
    transform_name: str,
    transform_params: dict,
    value_col: str,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict:
    """Fit one household; return metrics dict. Safe for joblib parallelism."""
    try:
        data  = pd.read_parquet(Path(features_dir_str), filters=[("LCLid", "in", [lclid])])
        tstp  = data.index.get_level_values("tstp")
        train = data.loc[tstp < train_end]
        val   = data.loc[(tstp >= val_start)  & (tstp < val_end)]
        test  = data.loc[(tstp >= test_start) & (tstp < test_end)]

        if len(train) < 500 or len(val) == 0 or len(test) == 0:
            return {"LCLid": lclid, "error": "insufficient data"}

        d = mase_denom(train[value_col])
        if d == 0 or np.isnan(d):
            return {"LCLid": lclid, "error": "zero MASE denominator"}

        transform = (
            DeseasonalisingTransform(**transform_params)
            if transform_name == "DeseasonalisingTransform" else None
        )
        mc = ModelConfig(
            estimator           = _build_estimator(model_family, model_params),
            model_name          = model_family,
            normalize           = True,
            fill_missing        = True,
            encode_categoricals = True,
            categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        )
        wrapper = MLForecast(mc, feature_config, missing_config, transform)
        wrapper.fit(train)

        vm = eval_metrics(val[value_col],  wrapper.predict(val),  d)
        tm = eval_metrics(test[value_col], wrapper.predict(test), d)

        return {
            "LCLid":    lclid,
            "val_mae":  vm["mae"],  "val_rmse":  vm["rmse"],  "val_mase":  vm["mase"],
            "test_mae": tm["mae"],  "test_rmse": tm["rmse"],  "test_mase": tm["mase"],
            "n_train":  len(train), "n_val":     len(val),    "n_test":    len(test),
        }

    except Exception as exc:
        return {"LCLid": lclid, "error": str(exc)}


# %%
# --- Step 4: Parallel fitting ------------------------------------------------

logger.info("Fitting %d households (n_jobs=%s) …", len(cohort_ids), N_JOBS)

raw_results = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(_fit_one_household)(
        lclid            = lclid,
        features_dir_str = str(FEATURES_DIR),
        feature_config   = FEATURE_CONFIG,
        missing_config   = MISSING_CONFIG,
        model_family     = BEST_CFG["family"],
        model_params     = BEST_CFG["params"],
        transform_name   = BEST_CFG["transform"],
        transform_params = BEST_CFG["transform_params"],
        value_col        = VALUE_COL,
        train_end        = TRAIN_END,
        val_start        = VAL_START,
        val_end          = VAL_END,
        test_start       = TEST_START,
        test_end         = TEST_END,
    )
    for lclid in cohort_ids
)

errors    = [r for r in raw_results if "error" in r]
successes = [r for r in raw_results if "error" not in r]
if errors:
    logger.warning("%d households failed: %s", len(errors),
                   [(e["LCLid"], e["error"]) for e in errors])

per_hh = pd.DataFrame(successes).set_index("LCLid")
logger.info("Fitted: %d / %d households", len(per_hh), len(cohort_ids))
save_csv(per_hh, "task07_per_household_metrics.csv", ARTIFACTS_DIR)

# %%
# --- Step 5: Summary statistics ----------------------------------------------

metric_cols = ["val_mae", "val_rmse", "val_mase", "test_mae", "test_rmse", "test_mase"]
summary = per_hh[metric_cols].agg(["mean", "median", "std"]).T
summary.columns = ["mean", "median", "std"]
save_csv(summary, "task07_cohort_summary.csv", ARTIFACTS_DIR)
logger.info("Cohort summary:\n%s", summary.to_string())

# %%
# --- Step 6: Violin plot — Test MASE -----------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))
parts = ax.violinplot(per_hh["test_mase"].dropna().values,
                      positions=[0], showmedians=True, showextrema=True)
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue"); pc.set_alpha(0.6)
ax.scatter(np.zeros(len(per_hh)), per_hh["test_mase"].values,
           s=10, color="navy", alpha=0.4, zorder=3)
ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1, label="MASE = 1 (naïve)")
ax.set_xticks([0]); ax.set_xticklabels([BEST_CFG["family"]])
ax.set_ylabel("Test MASE")
ax.set_title(f"Test MASE distribution — {len(per_hh)} households")
ax.legend(); fig.tight_layout()
save_fig(fig, "task07_test_mase_violin.png", ARTIFACTS_DIR)

# %%
# --- Step 7: Hard households investigation -----------------------------------

n_hard   = max(2, int(np.ceil(len(per_hh) * 0.10)))
hard_ids = per_hh["test_mase"].nlargest(n_hard).index.tolist()
easy_ids = per_hh["test_mase"].nsmallest(n_hard).index.tolist()

logger.info("Hard households (top 10%%, n=%d): %s", n_hard, hard_ids)
logger.info("Hard vs Easy — Test MASE: hard=%.3f  easy=%.3f",
            per_hh.loc[hard_ids, "test_mase"].mean(),
            per_hh.loc[easy_ids, "test_mase"].mean())

# Load quality screen to annotate hard households
quality: pd.DataFrame | None = None
_quality_path = ARTIFACTS_DIR / "task04_household_quality_good.csv"
if _quality_path.exists():
    quality = pd.read_csv(_quality_path, index_col="LCLid")

hard_detail = per_hh.loc[hard_ids, metric_cols]
if quality is not None:
    hard_detail = hard_detail.join(quality[["imputed_frac", "max_zero_run", "mean_energy"]])
save_csv(hard_detail, "task07_hard_households_detail.csv", ARTIFACTS_DIR)

# Plot time series for 2 hard + 2 easy households
_plot_ids = hard_ids[:2] + easy_ids[:2]
_labels   = ["Hard"] * 2 + ["Easy"] * 2
_colors   = ["crimson", "tomato", "steelblue", "cornflowerblue"]

fig, axes = plt.subplots(len(_plot_ids), 1, figsize=(14, 3 * len(_plot_ids)), sharex=False)
for ax, lclid, label, color in zip(axes, _plot_ids, _labels, _colors):
    _s = pd.read_parquet(IMPUTED_PATH, filters=[("LCLid", "in", [lclid])],
                         columns=[VALUE_COL]).xs(lclid, level="LCLid")[VALUE_COL]
    ax.plot(_s.index, _s.values, linewidth=0.5, color=color)
    ax.axvline(TRAIN_END,  color="darkorange", linewidth=1, linestyle="--")
    ax.axvline(TEST_START, color="purple",     linewidth=1, linestyle="--")
    ax.set_title(f"{label}: {lclid}  (Test MASE={per_hh.loc[lclid, 'test_mase']:.3f})", fontsize=9)
    ax.set_ylabel("kWh/hh"); ax.tick_params(axis="x", labelrotation=20)
fig.suptitle("Hard vs Easy households — consumption patterns", fontsize=11)
fig.tight_layout()
save_fig(fig, "task07_hard_vs_easy_series.png", ARTIFACTS_DIR)

# %%
# --- Step 8: Acorn subgroup comparison (Kruskal-Wallis) ----------------------

if acorn_df is not None:
    _joined = per_hh[["test_mase"]].join(acorn_df, how="inner")
    _groups = {g: grp["test_mase"].dropna().values
               for g, grp in _joined.groupby("Acorn_grouped") if len(grp) >= 3}
    if len(_groups) >= 2:
        _stat, _pval = scs.kruskal(*_groups.values())
        logger.info("Kruskal-Wallis: H=%.4f  p=%.4f  groups=%s",
                    _stat, _pval, list(_groups.keys()))
        fig, ax = plt.subplots(figsize=(max(6, len(_groups) * 1.5), 5))
        ax.boxplot(list(_groups.values()), labels=list(_groups.keys()),
                   patch_artist=True, medianprops={"color": "crimson", "linewidth": 1.5})
        ax.set_ylabel("Test MASE")
        ax.set_title(f"Test MASE by Acorn group  (KW p={_pval:.3f})", fontsize=10)
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        save_fig(fig, "task07_acorn_mase_boxplot.png", ARTIFACTS_DIR)
        save_csv(
            pd.DataFrame({g: {"n": len(v), "mean_mase": round(v.mean(), 4),
                               "median_mase": round(np.median(v), 4), "kw_p": round(_pval, 4)}
                          for g, v in _groups.items()}).T,
            "task07_acorn_subgroup_stats.csv", ARTIFACTS_DIR,
        )
    else:
        logger.info("Fewer than 2 Acorn groups with ≥3 households — skipping KW test.")

# %%
# --- Step 9: Cohort interpretation -------------------------------------------

_median_mase    = per_hh["test_mase"].median()
_std_mase       = per_hh["test_mase"].std()
_pct_beat_naive = (per_hh["test_mase"] < 1.0).mean() * 100

logger.info(
    "Median Test MASE=%.3f  Std=%.3f  — %.0f%% of households beat naïve baseline.",
    _median_mase, _std_mase, _pct_beat_naive,
)
if _std_mase > 0.3:
    logger.info("High cross-household variance suggests heterogeneity — global model may help.")
if per_hh["test_mase"].max() > 2.0:
    logger.info("Some households MASE > 2 — consider excluding from pooled training.")

"""
Task 10: Hyperparameter Tuning of the Global Model

Tuning objective: Validation MAE (consistent across all three strategies).

Pipeline:
  1. Load best encoding variant from Task 9 and rebuild pooled datasets.
  2. Strategy 1 — Grid search       (3×3×3 = 27 configs).
  3. Strategy 2 — Random search     (≥27 configs).
  4. Strategy 3 — Bayesian (Optuna) (≥27 trials).
  5. Each strategy: select best by val MAE, evaluate test once after selection.
  6. Final comparison table (7 rows: naïve → tuned GFMs).

Validation checks:
  (a) All strategies optimise the same val objective (Val MAE).
  (b) Test split used exactly once per strategy, after val-based selection.
  (c) random_state=42 throughout for reproducibility.
  (d) Wall-clock time reported per strategy.
"""

# %%
import json
import logging
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    VALUE_COL,
    make_feature_config, make_missing_config,
    mase_denom, eval_metrics, per_hh_metrics,
    load_cohort, load_splits, add_lclid_enc,
    save_fig, save_csv,
)
from src.configs import ModelConfig
from src.forecasting import MLForecast
from src.transforms.transforms import DeseasonalisingTransform
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder

# %%

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "results" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Run mode ──────────────────────────────────────────────────────────────────
# QUICK_RUN=True  → 5 households, 5 random trials, 5 Optuna trials; fast.
# QUICK_RUN=False → 50 households, 27 random trials, 27 Optuna trials.
QUICK_RUN        = True
MAX_HOUSEHOLDS   = 5  if QUICK_RUN else 50
RANDOM_SEARCH_N  = 5  if QUICK_RUN else 27
OPTUNA_N_TRIALS  = 5  if QUICK_RUN else 27
RANDOM_STATE     = 42

# Fixed LightGBM params (not part of the search)
_FIXED = dict(n_estimators=500, learning_rate=0.05, min_child_samples=20, subsample=0.8)

# %%
# --- Step 1: Load Task 9 best variant + rebuild datasets ---------------------

_best_path = ARTIFACTS_DIR / "task09_best_variant.json"
if not _best_path.exists():
    raise FileNotFoundError(f"Task 09 best variant not found at {_best_path}. Run task09 first.")

with open(_best_path) as f:
    BEST9 = json.load(f)
logger.info("Task 09 best variant: %s  (Val MAE=%.4f)", BEST9["variant"], BEST9["val_mae"])

META_COLS = BEST9["meta_cols"]
variant   = BEST9["variant"]

# Load cohort + splits
cohort_ids = load_cohort(ARTIFACTS_DIR, MAX_HOUSEHOLDS)
train_pool, val_pool, test_pool = load_splits(FEATURES_DIR, cohort_ids)
add_lclid_enc(train_pool, val_pool, test_pool, cohort_ids)

# Reconstruct the best encoding variant
_acorn_path = PROJECT_ROOT / "data" / "london_smart_meters" / "informations_households.csv"
meta = (
    pd.read_csv(_acorn_path, usecols=["LCLid"] + META_COLS)
    .set_index("LCLid")
    .loc[lambda df: ~df.index.duplicated(keep="first")]
    .fillna("Unknown")
    .reindex(cohort_ids)
    .fillna("Unknown")
)


def _join_meta(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    lclids = pool.index.get_level_values("LCLid")
    for col in META_COLS:
        pool[col] = lclids.map(meta[col]).values
    return pool


if variant == "a_baseline":
    train_tune, val_tune, test_tune = train_pool, val_pool, test_pool
    extra_continuous, extra_categorical = [], []

elif variant == "b_ohe":
    train_tune = _join_meta(train_pool)
    val_tune   = _join_meta(val_pool)
    test_tune  = _join_meta(test_pool)
    extra_continuous, extra_categorical = [], META_COLS

elif variant == "c_count":
    def _count_encode(tr, va, te):
        tr, va, te = tr.copy(), va.copy(), te.copy()
        for col in META_COLS:
            counts = tr[col].value_counts()
            enc    = f"{col}_cnt"
            for p in (tr, va, te):
                p[enc] = p[col].map(counts).fillna(0).astype(float)
                p.drop(columns=[col], inplace=True)
        return tr, va, te
    train_tune, val_tune, test_tune = _count_encode(
        _join_meta(train_pool), _join_meta(val_pool), _join_meta(test_pool))
    extra_continuous  = [f"{c}_cnt" for c in META_COLS]
    extra_categorical = []

else:  # d_target_enc
    def _target_encode(tr, va, te):
        tr, va, te = tr.copy(), va.copy(), te.copy()
        g_mean = tr[VALUE_COL].mean()
        for col in META_COLS:
            means = tr.groupby(col)[VALUE_COL].mean()
            enc   = f"{col}_te"
            for p in (tr, va, te):
                p[enc] = p[col].map(means).fillna(g_mean).astype(float)
                p.drop(columns=[col], inplace=True)
        return tr, va, te
    train_tune, val_tune, test_tune = _target_encode(
        _join_meta(train_pool), _join_meta(val_pool), _join_meta(test_pool))
    extra_continuous  = [f"{c}_te" for c in META_COLS]
    extra_categorical = []

logger.info("Tuning datasets ready — train: %d  val: %d  test: %d",
            len(train_tune), len(val_tune), len(test_tune))

# %%
# --- Step 2: Shared config + helpers -----------------------------------------

FEATURE_CONFIG = make_feature_config(
    native_categorical = True,
    extra_continuous   = extra_continuous,
    extra_categorical  = extra_categorical,
)
MISSING_CONFIG = make_missing_config()


def _build_wrapper(search_params: dict) -> MLForecast:
    params = {**search_params, **_FIXED, "random_state": RANDOM_STATE, "verbose": -1}
    mc = ModelConfig(
        estimator           = LGBMRegressor(**params),
        model_name          = "LightGBM_tuned",
        normalize           = False,
        fill_missing        = True,
        encode_categoricals = True,
        categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    )
    return MLForecast(mc, FEATURE_CONFIG, MISSING_CONFIG, DeseasonalisingTransform(period=48))


def _val_mae(params: dict) -> float:
    w = _build_wrapper(params)
    w.fit(train_tune)
    y_hat  = w.predict(val_tune)
    y_true = val_tune[VALUE_COL].reindex(y_hat.index)
    return float((y_true - y_hat).abs().mean())


def _test_metrics_for(params: dict) -> dict:
    w = _build_wrapper(params)
    w.fit(train_tune)
    yv_hat = w.predict(val_tune)
    yt_hat = w.predict(test_tune)
    hh_df  = per_hh_metrics(yv_hat, yt_hat, val_tune, test_tune, train_tune, cohort_ids)
    return {
        "val_mae":   round(float(hh_df["val_mae"].mean()),   4),
        "val_mase":  round(float(hh_df["val_mase"].mean()),  4),
        "test_mae":  round(float(hh_df["test_mae"].mean()),  4),
        "test_mase": round(float(hh_df["test_mase"].mean()), 4),
    }


# %%
# --- Step 3: Grid search  (3×3×3 = 27 configs) --------------------------------
#
# Search space:
#   num_leaves       ∈ {16, 31, 63}
#   objective        ∈ {regression, regression_l1, huber}
#   colsample_bytree ∈ {0.5, 0.8, 1.0}

GRID = {
    "num_leaves":       [16, 31, 63],
    "objective":        ["regression", "regression_l1", "huber"],
    "colsample_bytree": [0.5, 0.8, 1.0],
}
n_grid = len(GRID["num_leaves"]) * len(GRID["objective"]) * len(GRID["colsample_bytree"])
logger.info("=== Strategy 1: Grid search — %d configs ===", n_grid)

grid_records = []
t0 = time.perf_counter()
for nl, obj, cbt in product(GRID["num_leaves"], GRID["objective"], GRID["colsample_bytree"]):
    p   = dict(num_leaves=nl, objective=obj, colsample_bytree=cbt)
    mae = _val_mae(p)
    grid_records.append({**p, "val_mae": mae})
    logger.info("  num_leaves=%d  obj=%-14s  cbt=%.1f  val_mae=%.4f", nl, obj, cbt, mae)
grid_elapsed = time.perf_counter() - t0

grid_df = pd.DataFrame(grid_records).sort_values("val_mae")
save_csv(grid_df.reset_index(drop=True), "task10_grid_search_results.csv", ARTIFACTS_DIR)

best_grid = {k: (int(v) if k == "num_leaves" else v)
             for k, v in grid_df.iloc[0].drop("val_mae").items()}
best_grid_val_mae = float(grid_df.iloc[0]["val_mae"])
logger.info("Grid best — val_mae=%.4f  params=%s  wall=%.1fs",
            best_grid_val_mae, best_grid, grid_elapsed)

grid_test = _test_metrics_for(best_grid)
logger.info("Grid test metrics: %s", grid_test)

# %%
# --- Step 4: Random search (≥27 configs) -------------------------------------
#
# Broader space (continuous):
#   num_leaves       ∈ [10, 100]    (integer)
#   objective        ∈ {regression, regression_l1, huber}
#   colsample_bytree ∈ [0.3, 1.0]
#   lambda_l1        ∈ [0, 10]
#   lambda_l2        ∈ [0, 10]

logger.info("=== Strategy 2: Random search — %d configs ===", RANDOM_SEARCH_N)
rng  = np.random.default_rng(RANDOM_STATE)
OBJS = ["regression", "regression_l1", "huber"]

rand_records = []
t0 = time.perf_counter()
for _ in range(RANDOM_SEARCH_N):
    p = {
        "num_leaves":       int(rng.integers(10, 101)),
        "objective":        OBJS[int(rng.integers(0, 3))],
        "colsample_bytree": float(rng.uniform(0.3, 1.0)),
        "lambda_l1":        float(rng.uniform(0, 10)),
        "lambda_l2":        float(rng.uniform(0, 10)),
    }
    mae = _val_mae(p)
    rand_records.append({**p, "val_mae": mae})
rand_elapsed = time.perf_counter() - t0

rand_df = pd.DataFrame(rand_records).sort_values("val_mae")
save_csv(rand_df.reset_index(drop=True), "task10_random_search_results.csv", ARTIFACTS_DIR)

best_rand = {k: (int(v) if k == "num_leaves" else v)
             for k, v in rand_df.iloc[0].drop("val_mae").items()}
best_rand_val_mae = float(rand_df.iloc[0]["val_mae"])
logger.info("Random best — val_mae=%.4f  params=%s  wall=%.1fs",
            best_rand_val_mae, best_rand, rand_elapsed)

rand_test = _test_metrics_for(best_rand)
logger.info("Random test metrics: %s", rand_test)

# %%
# --- Step 5: Bayesian optimisation (Optuna TPE) ------------------------------

logger.info("=== Strategy 3: Bayesian (Optuna) — %d trials ===", OPTUNA_N_TRIALS)


def _optuna_objective(trial: optuna.Trial) -> float:
    p = {
        "num_leaves":       trial.suggest_int("num_leaves", 10, 100),
        "objective":        trial.suggest_categorical("objective", OBJS),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "lambda_l1":        trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2":        trial.suggest_float("lambda_l2", 0.0, 10.0),
    }
    return _val_mae(p)


t0    = time.perf_counter()
study = optuna.create_study(
    direction  = "minimize",
    sampler    = optuna.samplers.TPESampler(seed=RANDOM_STATE),
    study_name = "task10_lgbm_tuning",
)
study.optimize(_optuna_objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)
optuna_elapsed = time.perf_counter() - t0

best_optuna     = {**study.best_params, **_FIXED}
best_optuna_val = study.best_value

logger.info("Optuna best — val_mae=%.4f  params=%s  wall=%.1fs",
            best_optuna_val, study.best_params, optuna_elapsed)

optuna_df = study.trials_dataframe()[
    ["number", "value", "params_num_leaves", "params_objective",
     "params_colsample_bytree", "params_lambda_l1", "params_lambda_l2"]
].rename(columns={"number": "trial", "value": "val_mae",
                   "params_num_leaves": "num_leaves", "params_objective": "objective",
                   "params_colsample_bytree": "colsample_bytree",
                   "params_lambda_l1": "lambda_l1", "params_lambda_l2": "lambda_l2"})
save_csv(optuna_df.sort_values("val_mae").reset_index(drop=True),
         "task10_optuna_trials.csv", ARTIFACTS_DIR)

optuna_test = _test_metrics_for(best_optuna)
logger.info("Optuna test metrics: %s", optuna_test)

# %%
# --- Step 6: Optuna diagnostic plots -----------------------------------------

fig = optuna.visualization.matplotlib.plot_optimization_history(study)
fig.set_size_inches(8, 4)
fig.suptitle("Optuna — Optimisation History (Val MAE)", fontsize=11)
fig.tight_layout()
save_fig(fig, "task10_optuna_history.png", ARTIFACTS_DIR)

fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
fig2.set_size_inches(7, 4)
fig2.suptitle("Optuna — Hyperparameter Importance", fontsize=11)
fig2.tight_layout()
save_fig(fig2, "task10_optuna_importance.png", ARTIFACTS_DIR)

# %%
# --- Step 7: Load baseline metrics from earlier tasks ------------------------

def _load_metric(path: Path, col: str, agg="mean") -> float:
    if not path.exists():
        return np.nan
    df = pd.read_csv(path, index_col=0)
    if col in df.columns:
        return float(df[col].agg(agg))
    return np.nan


_t07 = ARTIFACTS_DIR / "task07_per_household_metrics.csv"
_t08 = ARTIFACTS_DIR / "task08_per_household_metrics.csv"

# Val MASE for tuned strategies (reuse the wrapper already fitted above)
def _val_mase_for(params: dict) -> float:
    w = _build_wrapper(params)
    w.fit(train_tune)
    hh = per_hh_metrics(w.predict(val_tune), w.predict(test_tune),
                        val_tune, test_tune, train_tune, cohort_ids)
    return float(hh["val_mase"].mean())


grid_val_mase   = _val_mase_for(best_grid)
rand_val_mase   = _val_mase_for(best_rand)
optuna_val_mase = _val_mase_for(best_optuna)

# %%
# --- Step 8: Final comparison table ------------------------------------------

summary_rows = [
    {"model": "Seasonal naïve",
     "val_mae": np.nan, "val_mase": 1.0, "test_mae": np.nan, "test_mase": 1.0},
    {"model": "Best local model (Task 6–7)",
     "val_mae":  _load_metric(_t07, "val_mae"),
     "val_mase": _load_metric(_t07, "val_mase"),
     "test_mae": _load_metric(_t07, "test_mae"),
     "test_mase":_load_metric(_t07, "test_mase")},
    {"model": "GFM without meta-features (Task 8)",
     "val_mae":  _load_metric(_t08, "val_mae"),
     "val_mase": _load_metric(_t08, "val_mase"),
     "test_mae": _load_metric(_t08, "test_mae"),
     "test_mase":_load_metric(_t08, "test_mase")},
    {"model": f"GFM best encoding — {variant} (Task 9)",
     "val_mae":  BEST9["val_mae"], "val_mase":  BEST9["val_mase"],
     "test_mae": BEST9["test_mae"],"test_mase": BEST9["test_mase"]},
    {"model": "Tuned GFM — Grid",
     "val_mae":   best_grid_val_mae,  "val_mase":   grid_val_mase,
     "test_mae":  grid_test["test_mae"],   "test_mase":  grid_test["test_mase"]},
    {"model": "Tuned GFM — Random",
     "val_mae":   best_rand_val_mae,  "val_mase":   rand_val_mase,
     "test_mae":  rand_test["test_mae"],   "test_mase":  rand_test["test_mase"]},
    {"model": "Tuned GFM — Bayesian (Optuna)",
     "val_mae":   best_optuna_val,   "val_mase":   optuna_val_mase,
     "test_mae":  optuna_test["test_mae"],  "test_mase":  optuna_test["test_mase"]},
]

final_table = pd.DataFrame(summary_rows).set_index("model").round(4)
save_csv(final_table, "task10_final_comparison.csv", ARTIFACTS_DIR)
logger.info("Final comparison table:\n%s", final_table.to_string())

# %%
# --- Step 9: Comparison bar chart --------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric, title in zip(axes, ["val_mae", "test_mae"], ["Validation MAE", "Test MAE"]):
    vals = final_table[metric].dropna()
    bars = ax.barh(vals.index, vals.values, color="steelblue", edgecolor="k", linewidth=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_title(title, fontsize=11); ax.set_xlabel("MAE"); ax.invert_yaxis()
fig.suptitle("Task 10 — Final Model Comparison", fontsize=13)
fig.tight_layout()
save_fig(fig, "task10_final_comparison.png", ARTIFACTS_DIR)

# %%
# --- Step 10: Save best tuned params + tuning summary ------------------------

best_overall = min(
    [("grid",     best_grid,   best_grid_val_mae),
     ("random",   best_rand,   best_rand_val_mae),
     ("bayesian", best_optuna, best_optuna_val)],
    key=lambda x: x[2],
)
with open(ARTIFACTS_DIR / "task10_best_params.json", "w") as f:
    json.dump({"strategy": best_overall[0], "params": best_overall[1],
               "val_mae": best_overall[2]}, f, indent=2)

logger.info(
    """
Tuning summary (random_state=%d)
=================================
Strategy       configs  Val MAE   Wall time
Grid           %-6d   %.4f    %.1fs
Random         %-6d   %.4f    %.1fs
Bayesian       %-6d   %.4f    %.1fs

Best strategy: %s
Untuned GFM (Task 9) val_mae=%.4f — gain: %.4f
""",
    RANDOM_STATE,
    n_grid,         best_grid_val_mae,  grid_elapsed,
    RANDOM_SEARCH_N,best_rand_val_mae,  rand_elapsed,
    OPTUNA_N_TRIALS,best_optuna_val,    optuna_elapsed,
    best_overall[0],
    BEST9["val_mae"], BEST9["val_mae"] - best_overall[2],
)

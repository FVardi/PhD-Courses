"""
Task 11: Visual Inspection of Model Predictions

Fits five models and plots their predictions against the true time series
for a sample of households over the validation and test periods (plus the
last few days of training for context).

Models compared:
  (1) Seasonal naïve   — lag-48 of the raw series (no fitting)
  (2) Local LightGBM   — best config from Task 6, fitted per household
  (3) Global LightGBM  — pooled model, no meta-features (Task 8)
  (4) Global + meta    — best encoding variant from Task 9
  (5) Tuned global     — best hyperparameters from Task 10
"""

# %%
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    VALUE_COL, LGBM_DEFAULTS,
    make_feature_config, make_missing_config,
    load_cohort, load_splits, add_lclid_enc,
    save_fig,
)
from src.configs import ModelConfig
from src.forecasting import MLForecast
from src.transforms.transforms import ComposedTransform, DeseasonalisingTransform, LogTransform
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder

# %%

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "report" / "artifacts"
ACORN_PATH    = PROJECT_ROOT / "data" / "london_smart_meters" / "informations_households.csv"

QUICK_RUN      = False
MAX_HOUSEHOLDS = 5 if QUICK_RUN else 50
N_PLOT         = 3    # households to inspect
TRAIN_TAIL     = 48 * 3  # training days shown for context (3 days)
PLOT_DAYS      = 14      # days shown after the train/val boundary (None = show all)

TARGET_TRANSFORM = ComposedTransform([LogTransform(), DeseasonalisingTransform(period=48)])

# %%
# --- Step 1: Load saved configs ----------------------------------------------

with open(ARTIFACTS_DIR / "task06_best_model.json") as f:
    CFG6 = json.load(f)
with open(ARTIFACTS_DIR / "task09_best_variant.json") as f:
    CFG9 = json.load(f)
with open(ARTIFACTS_DIR / "task10_best_params.json") as f:
    CFG10 = json.load(f)

META_COLS = CFG9["meta_cols"]
variant   = CFG9["variant"]

# %%
# --- Step 2: Load cohort + splits --------------------------------------------

cohort_ids = load_cohort(ARTIFACTS_DIR, MAX_HOUSEHOLDS)
train_pool, val_pool, test_pool = load_splits(FEATURES_DIR, cohort_ids)
add_lclid_enc(train_pool, val_pool, test_pool, cohort_ids)

plot_ids = cohort_ids[:N_PLOT]

# %%
# --- Step 3: Meta-feature encoding -------------------------------------------

meta = (
    pd.read_csv(ACORN_PATH, usecols=["LCLid"] + META_COLS)
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


def _apply_variant(tr, va, te):
    """Return (train, val, test, extra_continuous, extra_categorical) for the saved variant."""
    if variant == "a_baseline":
        return tr, va, te, [], []
    if variant == "b_ohe":
        return _join_meta(tr), _join_meta(va), _join_meta(te), [], META_COLS
    if variant == "c_count":
        tr2, va2, te2 = tr.copy(), va.copy(), te.copy()
        for col in META_COLS:
            counts = tr[col].value_counts()
            enc = f"{col}_cnt"
            for p in (tr2, va2, te2):
                p[enc] = p[col].map(counts).fillna(0).astype(float)
                p.drop(columns=[col], inplace=True)
        return tr2, va2, te2, [f"{c}_cnt" for c in META_COLS], []
    # d_target_enc
    tr2, va2, te2 = tr.copy(), va.copy(), te.copy()
    g_mean = tr[VALUE_COL].mean()
    for col in META_COLS:
        means = tr.groupby(col)[VALUE_COL].mean()
        enc = f"{col}_te"
        for p in (tr2, va2, te2):
            p[enc] = p[col].map(means).fillna(g_mean).astype(float)
            p.drop(columns=[col], inplace=True)
    return tr2, va2, te2, [f"{c}_te" for c in META_COLS], []


# %%
# --- Step 4: Model builder helpers -------------------------------------------

def _make_wrapper(params: dict, extra_continuous=None, extra_categorical=None,
                  native_cat=True, normalize=False) -> MLForecast:
    mc = ModelConfig(
        estimator           = LGBMRegressor(**params),
        model_name          = "LightGBM",
        normalize           = normalize,
        fill_missing        = True,
        encode_categoricals = True,
        categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    )
    fc = make_feature_config(
        native_categorical = native_cat,
        extra_continuous   = extra_continuous or [],
        extra_categorical  = extra_categorical or [],
    )
    return MLForecast(mc, fc, make_missing_config(), TARGET_TRANSFORM)


# %%
# --- Step 5: Fit global models (once, shared across all plot households) -----

logger.info("Fitting global baseline (Task 8) …")
global_params = {**LGBM_DEFAULTS, "random_state": 42, "verbose": -1}
w_global = _make_wrapper(global_params)
w_global.fit(train_pool)

logger.info("Fitting global + meta (Task 9 / Task 10) …")
tr_m, va_m, te_m, ec_m, ecat_m = _apply_variant(train_pool, val_pool, test_pool)
w_meta = _make_wrapper(global_params, extra_continuous=ec_m, extra_categorical=ecat_m)
w_meta.fit(tr_m)

logger.info("Fitting tuned global (Task 10) …")
tuned_params = {**CFG10["params"], "random_state": 42, "verbose": -1}
w_tuned = _make_wrapper(tuned_params, extra_continuous=ec_m, extra_categorical=ecat_m)
w_tuned.fit(tr_m)

# %%
# --- Step 6: Helpers for per-household prediction and naïve baseline ---------

def _hh_pred(lclid: str, wrapper: MLForecast, pool: pd.DataFrame) -> pd.Series:
    mask = pool.index.get_level_values("LCLid") == lclid
    return wrapper.predict(pool.loc[mask])


def _naive(lclid: str, pool: pd.DataFrame) -> pd.Series:
    mask = pool.index.get_level_values("LCLid") == lclid
    s = pool.loc[mask, VALUE_COL]
    return s.shift(48).dropna()


# %%
# --- Step 7: Plot each household ---------------------------------------------

STYLE = {
    "True":              dict(color="black",      linewidth=1.4, linestyle="-",  zorder=5),
    "Naïve":             dict(color="grey",       linewidth=0.8, linestyle="--", alpha=0.8),
    "Local":             dict(color="tab:green",  linewidth=1.0, linestyle="-",  alpha=0.9),
    "Global":            dict(color="tab:blue",   linewidth=1.0, linestyle="-",  alpha=0.9),
    "Global+meta":       dict(color="tab:orange", linewidth=1.0, linestyle="-",  alpha=0.9),
    "Tuned global+meta": dict(color="tab:red",    linewidth=1.0, linestyle="-",  alpha=0.9),
}

for lclid in plot_ids:
    logger.info("Fitting local model and plotting %s …", lclid)

    # Local model — fitted on this household's training data only
    local_params = {**CFG6["params"], "random_state": 42, "verbose": -1}
    mask_tr = train_pool.index.get_level_values("LCLid") == lclid
    w_local = _make_wrapper(local_params, native_cat=False, normalize=True)
    w_local.fit(train_pool.loc[mask_tr])

    # True series: last TRAIN_TAIL training points + full val + test
    mask_va = val_pool.index.get_level_values("LCLid") == lclid
    mask_te = test_pool.index.get_level_values("LCLid") == lclid
    y_true = pd.concat([
        train_pool.loc[mask_tr, VALUE_COL].iloc[-TRAIN_TAIL:],
        val_pool.loc[mask_va, VALUE_COL],
        test_pool.loc[mask_te, VALUE_COL],
    ])

    # Predictions over val + test
    preds = {
        "Naïve":        pd.concat([_naive(lclid, val_pool),               _naive(lclid, test_pool)]),
        "Local":        pd.concat([_hh_pred(lclid, w_local,  val_pool),   _hh_pred(lclid, w_local,  test_pool)]),
        "Global":       pd.concat([_hh_pred(lclid, w_global, val_pool),   _hh_pred(lclid, w_global, test_pool)]),
        "Global+meta":  pd.concat([_hh_pred(lclid, w_meta,   va_m),       _hh_pred(lclid, w_meta,   te_m)]),
        "Tuned global+meta": pd.concat([_hh_pred(lclid, w_tuned,  va_m),  _hh_pred(lclid, w_tuned,  te_m)]),
    }

    # Optionally zoom in to PLOT_DAYS after the training boundary
    if PLOT_DAYS is not None:
        t_end = y_true.index.get_level_values("tstp")[TRAIN_TAIL + PLOT_DAYS * 48 - 1]
        y_true = y_true[y_true.index.get_level_values("tstp") <= t_end]
        preds  = {k: v[v.index.get_level_values("tstp") <= t_end] for k, v in preds.items()}

    fig, ax = plt.subplots(figsize=(16, 4))

    t_true = y_true.index.get_level_values("tstp")
    ax.plot(t_true, y_true.values, label="True", **STYLE["True"])
    for name, yhat in preds.items():
        t_hat = yhat.index.get_level_values("tstp")
        ax.plot(t_hat, yhat.values, label=name, **STYLE[name])

    # Pin x-axis to the plotted data range before adding vertical lines
    t_min, t_max = t_true[0], t_true[-1]
    ax.set_xlim(t_min, t_max)

    val_start  = val_pool.index.get_level_values("tstp").min()
    test_start = test_pool.index.get_level_values("tstp").min()
    if t_min <= val_start <= t_max:
        ax.axvline(val_start,  color="steelblue", linewidth=1, linestyle=":", label="Val start")
    if t_min <= test_start <= t_max:
        ax.axvline(test_start, color="purple",    linewidth=1, linestyle=":", label="Test start")

    ax.set_title(f"Household {lclid} — predictions vs true series", fontsize=10)
    ax.set_ylabel("Energy (kWh/hh)")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    fig.tight_layout()
    save_fig(fig, f"task11_predictions_{lclid}.png", ARTIFACTS_DIR)

logger.info("Task 11 complete. %d plots saved to %s", len(plot_ids), ARTIFACTS_DIR)

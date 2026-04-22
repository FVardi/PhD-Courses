"""
Task 06: Local Single-Household Forecasting Study

Pipeline:
  1. Select one representative household (Task 4 quality screen).
  2. Sweep linear models: LinearRegression, Ridge, Lasso.
  3. Sweep tree models: DecisionTree, RandomForest, LightGBM.
  4. Evaluate all models on validation and test splits.
  5. Select best by validation MAE; save config for Task 07.
  6. Residual analysis for the best model.

MASE denominator: mean |y_t − y_{t−48}| on training data only.
"""

# TODO: Apply transforsm from task 4. Probably log transform and de-seasoning.

# %%
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from statsmodels.graphics.tsaplots import plot_acf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    VALUE_COL, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    make_feature_config, make_missing_config, make_sklearn_model_config,
    mase_denom, eval_metrics,
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
# QUICK_RUN=True  → 2 sweep values per model; fast for inspection.
# QUICK_RUN=False → full sweeps (recommended before submitting).
QUICK_RUN = True

RIDGE_ALPHAS  = [1.0, 10.0]              if QUICK_RUN else [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
LASSO_ALPHAS  = [0.001, 0.01]            if QUICK_RUN else [0.0001, 0.001, 0.01, 0.1, 1.0]
DT_DEPTHS     = [5, 8]                   if QUICK_RUN else [3, 5, 8, 12]
RF_DEPTHS     = [5, None]                if QUICK_RUN else [5, 10, 20, None]
LGBM_LEAVES   = [31, 63]                 if QUICK_RUN else [15, 31, 63, 127]

TARGET_TRANSFORM = DeseasonalisingTransform(period=48)

# %%
# --- Step 1: Household selection ----------------------------------------------
#
# Prefer zero imputation + zero zero-run; break ties by most available slots.

_quality_path = ARTIFACTS_DIR / "task04_household_quality_good.csv"
if _quality_path.exists():
    _quality = pd.read_csv(_quality_path, index_col="LCLid")
    _best = (
        _quality
        .query("imputed_frac == 0 and max_zero_run == 0")
        .sort_values("n_slots", ascending=False)
    )
    LCLID = _best.index[0] if len(_best) else _quality.sort_values("imputed_frac").index[0]
    logger.info(
        "Selected household: %s  (imputed=%.2f%%  zero_run=%d  n_slots=%d)",
        LCLID,
        _quality.loc[LCLID, "imputed_frac"] * 100,
        _quality.loc[LCLID, "max_zero_run"],
        _quality.loc[LCLID, "n_slots"],
    )
else:
    LCLID = "MAC000002"
    logger.warning("Quality CSV not found — falling back to %s", LCLID)

# %%
# --- Step 2: Load and split --------------------------------------------------

data  = pd.read_parquet(FEATURES_DIR, filters=[("LCLid", "in", [LCLID])])
_tstp = data.index.get_level_values("tstp")
train = data.loc[_tstp < TRAIN_END]
val   = data.loc[(_tstp >= VAL_START) & (_tstp < VAL_END)]
test  = data.loc[(_tstp >= TEST_START) & (_tstp < TEST_END)]
logger.info("Split sizes — train: %d  val: %d  test: %d", len(train), len(val), len(test))

# %%
# --- Step 3: Shared configs --------------------------------------------------

feature_config = make_feature_config()
missing_config = make_missing_config()
_denom         = mase_denom(train[VALUE_COL])
logger.info("MASE denominator (lag-48 naïve, training): %.6f", _denom)

# ── Helpers ───────────────────────────────────────────────────────────────────

results_rows:    list[dict]           = []
fitted_models:   dict[str, MLForecast] = {}
_config_registry: dict                = {}


def _fit_eval(estimator, name: str) -> MLForecast:
    mc = make_sklearn_model_config(estimator, name)
    w  = MLForecast(mc, feature_config, missing_config, TARGET_TRANSFORM)
    w.fit(train)
    vm = eval_metrics(val[VALUE_COL],  w.predict(val),  _denom)
    tm = eval_metrics(test[VALUE_COL], w.predict(test), _denom)
    row = {"model": name,
           "val_mae":  vm["mae"],  "val_rmse":  vm["rmse"],  "val_mase":  vm["mase"],
           "test_mae": tm["mae"],  "test_rmse": tm["rmse"],  "test_mase": tm["mase"]}
    logger.info("%-35s | Val MAE=%.4f MASE=%.4f | Test MAE=%.4f MASE=%.4f",
                name, vm["mae"], vm["mase"], tm["mae"], tm["mase"])
    results_rows.append(row)
    return w


def _sweep_val_mae(estimator, name: str) -> float:
    mc = make_sklearn_model_config(estimator, name)
    w  = MLForecast(mc, feature_config, missing_config, TARGET_TRANSFORM)
    w.fit(train)
    return eval_metrics(val[VALUE_COL], w.predict(val), _denom)["mae"]


# %%
# --- Step 4: Linear models ---------------------------------------------------

w = _fit_eval(LinearRegression(), "LinearRegression")
fitted_models["LinearRegression"] = w
_config_registry["LinearRegression"] = {"family": "LinearRegression", "params": {}}

logger.info("Ridge alpha sweep …")
_best_alpha = min(RIDGE_ALPHAS, key=lambda a: _sweep_val_mae(Ridge(alpha=a), f"Ridge(a={a})"))
logger.info("Best Ridge alpha: %s", _best_alpha)
w = _fit_eval(Ridge(alpha=_best_alpha), f"Ridge(a={_best_alpha})")
fitted_models["Ridge"] = w
_config_registry[f"Ridge(a={_best_alpha})"] = {"family": "Ridge", "params": {"alpha": _best_alpha}}

logger.info("Lasso alpha sweep …")
_best_la = min(LASSO_ALPHAS, key=lambda a: _sweep_val_mae(Lasso(alpha=a, max_iter=5000), f"Lasso(a={a})"))
logger.info("Best Lasso alpha: %s", _best_la)
w = _fit_eval(Lasso(alpha=_best_la, max_iter=5000), f"Lasso(a={_best_la})")
fitted_models["Lasso"] = w
_config_registry[f"Lasso(a={_best_la})"] = {"family": "Lasso", "params": {"alpha": _best_la, "max_iter": 5000}}

# %%
# --- Step 5: Tree models -----------------------------------------------------

logger.info("DecisionTree depth sweep …")
_best_dt = min(DT_DEPTHS, key=lambda d: _sweep_val_mae(
    DecisionTreeRegressor(max_depth=d, random_state=42), f"DT(d={d})"))
logger.info("Best DT depth: %s", _best_dt)
w = _fit_eval(DecisionTreeRegressor(max_depth=_best_dt, random_state=42), f"DecisionTree(d={_best_dt})")
fitted_models["DecisionTree"] = w
_config_registry[f"DecisionTree(d={_best_dt})"] = {
    "family": "DecisionTree", "params": {"max_depth": _best_dt, "random_state": 42}}

logger.info("RandomForest depth sweep …")
_best_rf = min(RF_DEPTHS, key=lambda d: _sweep_val_mae(
    RandomForestRegressor(max_depth=d, n_estimators=100, random_state=42, n_jobs=-1), f"RF(d={d})"))
logger.info("Best RF depth: %s", _best_rf)
w = _fit_eval(
    RandomForestRegressor(max_depth=_best_rf, n_estimators=100, random_state=42, n_jobs=-1),
    f"RandomForest(d={_best_rf})")
fitted_models["RandomForest"] = w
_config_registry[f"RandomForest(d={_best_rf})"] = {
    "family": "RandomForest",
    "params": {"max_depth": _best_rf, "n_estimators": 100, "random_state": 42}}

logger.info("LightGBM leaves sweep …")
_best_lgbm = min(LGBM_LEAVES, key=lambda l: _sweep_val_mae(
    LGBMRegressor(num_leaves=l, n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1),
    f"LGBM(l={l})"))
logger.info("Best LGBM leaves: %s", _best_lgbm)
w = _fit_eval(
    LGBMRegressor(num_leaves=_best_lgbm, n_estimators=200, learning_rate=0.05,
                  random_state=42, verbose=-1),
    f"LightGBM(l={_best_lgbm})")
fitted_models["LightGBM"] = w
_config_registry[f"LightGBM(l={_best_lgbm})"] = {
    "family": "LightGBM",
    "params": {"num_leaves": _best_lgbm, "n_estimators": 200,
               "learning_rate": 0.05, "random_state": 42, "verbose": -1}}

# %%
# --- Step 6: Comparison table ------------------------------------------------

comparison = pd.DataFrame(results_rows).set_index("model")
save_csv(comparison, "task06_model_comparison.csv", ARTIFACTS_DIR)
logger.info("Model comparison:\n%s", comparison.to_string())

# %%
# --- Step 7: Best model selection + save config for Task 07 ------------------

best_name    = comparison["val_mae"].idxmin()
best_wrapper = fitted_models[next(k for k in fitted_models if best_name.startswith(k))]

logger.info(
    "Best model: %s  (Val MAE=%.4f  Val MASE=%.4f)",
    best_name, comparison.loc[best_name, "val_mae"], comparison.loc[best_name, "val_mase"],
)

_best_cfg = {
    **_config_registry[best_name],
    "model_name":       best_name,
    "val_mae":          float(comparison.loc[best_name, "val_mae"]),
    "transform":        "DeseasonalisingTransform",
    "transform_params": {"period": 48},
}
_cfg_path = ARTIFACTS_DIR / "task06_best_model.json"
with open(_cfg_path, "w") as _f:
    json.dump(_best_cfg, _f, indent=2)
logger.info("Best model config saved → %s", _cfg_path)

# %%
# --- Step 8: Residual analysis (test set) ------------------------------------

y_hat_test  = best_wrapper.predict(test)
y_true_test = test[VALUE_COL].reindex(y_hat_test.index)
residuals   = (y_true_test - y_hat_test).dropna()
tstp_idx    = residuals.index.get_level_values("tstp")
_hour       = pd.Series(tstp_idx.hour, index=residuals.index)
_mae_hr     = residuals.abs().groupby(_hour).mean()

fig, axes = plt.subplots(2, 3, figsize=(18, 9))

axes[0, 0].plot(tstp_idx, residuals.values, linewidth=0.4, color="steelblue")
axes[0, 0].axhline(0, color="crimson", linewidth=0.8, linestyle="--")
axes[0, 0].set_title("Residuals vs Time")
axes[0, 0].set_xlabel("Date"); axes[0, 0].set_ylabel("Residual (kWh/hh)")
axes[0, 0].tick_params(axis="x", labelrotation=30)

axes[0, 1].hist(residuals.values, bins=80, color="steelblue", edgecolor="none")
axes[0, 1].axvline(0, color="crimson", linewidth=0.8, linestyle="--")
axes[0, 1].set_title("Residual Distribution")
axes[0, 1].set_xlabel("Residual (kWh/hh)"); axes[0, 1].set_ylabel("Count")

(osm, osr), (slope, intercept, _) = stats.probplot(residuals.values, dist="norm")
axes[0, 2].scatter(osm, osr, s=2, alpha=0.4, color="steelblue")
axes[0, 2].plot(osm, slope * np.array(osm) + intercept, color="crimson", linewidth=1)
axes[0, 2].set_title("Q–Q Plot (Normal)")
axes[0, 2].set_xlabel("Theoretical quantiles"); axes[0, 2].set_ylabel("Sample quantiles")

plot_acf(residuals.values, lags=96, ax=axes[1, 0], title="ACF of Residuals")
axes[1, 0].set_xlabel("Lag (half-hours)")

axes[1, 1].bar(_mae_hr.index, _mae_hr.values, color="steelblue", width=0.7)
axes[1, 1].set_title("MAE by Hour of Day")
axes[1, 1].set_xlabel("Hour"); axes[1, 1].set_ylabel("MAE (kWh/hh)")
axes[1, 1].set_xticks(range(0, 24, 2))

axes[1, 2].scatter(y_true_test.values, y_hat_test.reindex(y_true_test.index).values,
                   s=1, alpha=0.2, color="steelblue")
_lim = (min(y_true_test.min(), y_hat_test.min()), max(y_true_test.max(), y_hat_test.max()))
axes[1, 2].plot(_lim, _lim, color="crimson", linewidth=1)
axes[1, 2].set_title("Predicted vs Actual")
axes[1, 2].set_xlabel("Actual (kWh/hh)"); axes[1, 2].set_ylabel("Predicted (kWh/hh)")

fig.suptitle(f"Residual Analysis — {best_name}  ({LCLID}, test set)", fontsize=12)
fig.tight_layout()
save_fig(fig, "task06_residual_analysis.png", ARTIFACTS_DIR)

# %%
# --- Step 9: Residual interpretation -----------------------------------------

_skewness  = float(pd.Series(residuals.values).skew())
_kurt      = float(pd.Series(residuals.values).kurt())
_acf_lag1  = float(residuals.autocorr(lag=1))
_acf_lag48 = float(residuals.autocorr(lag=48))
_peak_hour = int(_mae_hr.idxmax())

logger.info("Residual diagnostics (test set):")
logger.info("  Skewness:        %.3f", _skewness)
logger.info("  Kurtosis:        %.3f", _kurt)
logger.info("  ACF lag-1:       %.3f", _acf_lag1)
logger.info("  ACF lag-48:      %.3f", _acf_lag48)
logger.info("  Peak-error hour: %02d:00", _peak_hour)

if abs(_acf_lag48) > 0.1:
    logger.info("  → Residual ACF at lag-48 non-trivial: deseasonalisation incomplete.")
if abs(_acf_lag1) > 0.1:
    logger.info("  → Positive lag-1 autocorrelation: consider AR(1) correction.")
if _skewness > 0.5:
    logger.info("  → Right-skewed residuals: log/Box-Cox transform may reduce heteroscedasticity.")

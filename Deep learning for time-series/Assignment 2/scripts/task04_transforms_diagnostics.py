"""
Task 04: Transforms & Diagnostics

Stationarity diagnostics and OLS baseline for half-hourly energy data.
All reusable logic lives in src/transforms/ (diagnostics, ols) and
src/evaluation/plots.py.

Outputs (plots as PNG, tables as CSV) are written to results/artifacts/.
"""

# %%
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plots import plot_acf_pacf, plot_households_with_splits, plot_ols_diagnostics
from src.transforms.diagnostics import run_adf_tests, run_mann_kendall, run_white_test
from src.transforms.ols import fit_ols_households
from src.transforms.transforms import (
    DeseasonalisingTransform,
    DetrendingTransform,
    LogTransform,
)

# %%

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "results" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

VALUE_COL = "energy_imputed_seasonal"
LCLIDS    = ["MAC000002", "MAC000003", "MAC000004", "MAC000005", "MAC000006"]

# --- Data splits --------------------------------------------------------------
# All diagnostics, transforms, and modelling must respect these boundaries.
TRAIN_END  = pd.Timestamp("2014-01-01")   # train: strictly before this date
VAL_START  = pd.Timestamp("2014-01-01")
VAL_END    = pd.Timestamp("2014-02-01")   # val:  [2014-01-01, 2014-02-01)
TEST_START = pd.Timestamp("2014-02-01")
TEST_END   = pd.Timestamp("2014-03-01")   # test: [2014-02-01, 2014-03-01)


def _save_fig(fig: plt.Figure, name: str) -> None:
    path = ARTIFACTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", path)


def _save_csv(df: pd.DataFrame, name: str) -> None:
    path = ARTIFACTS_DIR / name
    df.to_csv(path)
    logger.info("Saved table  → %s", path)

# TODO: Implement check for time series quality again. It is needed here and used later in task 06. Include a check where the validation and test sets must have an average energy consumption above 0.25, since it doesnt make sense to predict intervals with no energy usage.

# %%
# ==============================================================================
# PART 1 — DIAGNOSTICS
# ==============================================================================

# %%
# --- Load target series for stationarity tests --------------------------------

logger.info("Loading %d households …", len(LCLIDS))

multi = pd.read_parquet(
    FEATURES_DIR,
    filters=[("LCLid", "in", LCLIDS)],
    columns=[VALUE_COL, "is_imputed", "gap_length"],
)

# TODO: Write to logger the max gap length and the imputation percentage to justify using these houses.
max_gap_length = multi["gap_length"].max()
imputation_percentage = multi["is_imputed"].mean() * 100 
logger.info("Max gap length: %d", max_gap_length)
logger.info("Imputation percentage: %.2f%%", imputation_percentage)


# %%
# --- ADF stationarity tests ---------------------------------------------------

adf_results = run_adf_tests(multi, LCLIDS, VALUE_COL, TRAIN_END)
_save_csv(adf_results, "task04_adf_results.csv")

n_stat = adf_results["stationary (p<0.05)"].sum()
logger.info(
    "ADF summary: %d / %d tests reject the unit-root null — series are stationary in a stochastic sense.",
    n_stat, len(adf_results),
)

# %%
# --- Mann-Kendall trend test --------------------------------------------------

mk_results = run_mann_kendall(multi, LCLIDS, VALUE_COL, TRAIN_END)
_save_csv(mk_results, "task04_mk_results.csv")

logger.info(
    "Mann-Kendall caveat: with %d training slots the test has very high statistical power — "
    "even a negligibly small trend will produce a significant p-value.  "
    "Visual inspection of the series is needed to judge practical significance.  "
    "Because the series is periodic, the choice of training window can also induce a spurious "
    "trend if the split does not fall at a natural cycle boundary; a significant result may "
    "not generalise to other time periods.",
    multi.loc[multi.index.get_level_values("tstp") < TRAIN_END].shape[0] // len(LCLIDS),
)

# %%
# --- Visual inspection --------------------------------------------------------

fig_ts = plot_households_with_splits(
    multi, LCLIDS, VALUE_COL,
    train_end=TRAIN_END, test_start=TEST_START,
)
_save_fig(fig_ts, "task04_households_with_splits.png")

# %%
# --- ACF / PACF up to lag 3×336 (3 weekly cycles) ----------------------------

train_flat = (
    multi
    .loc[multi.index.get_level_values("tstp") < TRAIN_END]
    .reset_index()
)

fig_acf = plot_acf_pacf(
    train_flat,
    household_ids=LCLIDS,
    lags=3 * 336,
    timestamp_col="tstp",
    value_col=VALUE_COL,
)
_save_fig(fig_acf, "task04_acf_pacf.png")

# %%
# --- OLS baseline -------------------------------------------------------------

ols_data = pd.read_parquet(
    FEATURES_DIR,
    filters=[("LCLid", "in", LCLIDS)],
)

metrics_list, results_list = fit_ols_households(ols_data, LCLIDS, VALUE_COL, TRAIN_END)

fig_ols = plot_ols_diagnostics(results_list)
_save_fig(fig_ols, "task04_ols_diagnostics.png")

ols_metrics_df = pd.DataFrame(metrics_list).set_index("household")
_save_csv(ols_metrics_df, "task04_ols_metrics.csv")

# %%
# --- White's heteroskedasticity test ------------------------------------------

white_results = run_white_test(results_list)
_save_csv(white_results, "task04_white_test.csv")

n_hetero = (white_results["LM p-value"] < 0.05).sum()
logger.info(
    "White test summary: %d / %d households show significant heteroskedasticity (p < 0.05) — "
    "a variance-stabilising transform (e.g. log, Box-Cox) may be warranted.",
    n_hetero, len(white_results),
)

# %%
# --- Diagnostic summary table -------------------------------------------------

def _recommend(row: pd.Series) -> str:
    recs = []
    if row["White p"] < 0.05:
        recs.append("log-transform")
    if row["trend slope"] != 0.0:
        recs.append("detrend")
    if row["ACF lag-48"] > 0.3 or row["ACF lag-336"] > 0.3:
        recs.append("seasonal features or deseasonalise")
    return ", ".join(recs) if recs else "none"

adf_pivot = (
    adf_results["p-value"]
    .unstack("regression")
    .rename(columns={"c": "ADF(c) p", "ct": "ADF(ct) p", "ctt": "ADF(ctt) p"})
)

acf_rows = {}
for lclid in LCLIDS:
    s = (
        multi.xs(lclid, level="LCLid")[VALUE_COL]
        .loc[lambda x: x.index < TRAIN_END]
        .dropna()
    )
    vals = acf(s, nlags=336, fft=True)
    acf_rows[lclid] = {"ACF lag-48": round(vals[48], 4), "ACF lag-336": round(vals[336], 4)}

acf_df = pd.DataFrame(acf_rows).T
acf_df.index.name = "household"

summary = (
    adf_pivot
    .join(mk_results[["slope"]].rename(columns={"slope": "trend slope"}))
    .join(acf_df)
    .join(white_results[["LM p-value"]].rename(columns={"LM p-value": "White p"}))
)
summary["heteroskedastic"] = summary["White p"] < 0.05
summary["recommended transform(s)"] = summary.apply(_recommend, axis=1)

_save_csv(summary, "task04_diagnostic_summary.csv")

# %%
# ==============================================================================
# PART 2 — TRANSFORMS
# ==============================================================================

# %%
# --- Available transforms -----------------------------------------------------
#
# DetrendingTransform(degree, per_household)
#   Fits a polynomial trend on training data and subtracts it.
#
# DeseasonalisingTransform(period)
#   Fits per-slot means (0 … period-1) on training data and subtracts them.
#
# LogTransform / BoxCoxTransform
#   Variance-stabilising; fitted on training data only.
#   Predictions are always inverse-transformed before computing metrics.
#
# All transforms are applied in the comparison experiment below.

# %%
# --- Transform comparison experiment ------------------------------------------
# One household, OLS model, validation set.  Only the target is transformed;
# the feature matrix (lags, rolling, EWMA, calendar, Fourier) is unchanged.
# All transforms are fitted on training data only; predictions are
# inverse-transformed to the original scale before metrics are computed.
#
# MASE denominator: mean |y_t - y_{t-48}| on training (seasonal naïve baseline).

EVAL_HOUSEHOLD = LCLIDS[0]

_hh      = ols_data.xs(EVAL_HOUSEHOLD, level="LCLid")
_tstp    = _hh.index
_train   = _hh.loc[_tstp < TRAIN_END]
_val     = _hh.loc[(_tstp >= VAL_START) & (_tstp < VAL_END)]

y_train  = _train[VALUE_COL].astype(float)
y_val    = _val[VALUE_COL].astype(float)

# Drop target and all other non-feature columns from X
_drop = [c for c in _hh.columns if c.startswith("energy") or c in ("gap_length", "is_imputed")]
X_train = _train.drop(columns=_drop).astype(float)
X_val   = _val.drop(columns=_drop).astype(float)


def _fit_predict(y_t: pd.Series, X_t: pd.DataFrame, X_v: pd.DataFrame) -> pd.Series:
    idx  = X_t.dropna(how="any").index.intersection(y_t.dropna().index)
    Xc_t = sm.add_constant(X_t.loc[idx], has_constant="add")
    Xc_v = sm.add_constant(X_v.dropna(how="any"), has_constant="add")
    return pd.Series(
        sm.OLS(y_t.loc[idx], Xc_t).fit().predict(Xc_v),
        index=Xc_v.index,
    )


_mase_denom = y_train.diff(48).abs().dropna().mean()


def _eval(y_actual: pd.Series, y_pred: pd.Series) -> dict:
    idx = y_actual.index.intersection(y_pred.index)
    mae = (y_actual.loc[idx] - y_pred.loc[idx]).abs().mean()
    return {"MAE": round(mae, 4), "MASE": round(mae / _mase_denom, 4)}


rows_cmp = []

# (a) No transform
rows_cmp.append({
    "condition": "(a) No transform",
    **_eval(y_val, _fit_predict(y_train, X_train, X_val)),
})

# (b) Differencing (lag=1) — reconstruct from last training value
_y_b  = y_train.diff(1).dropna()
_yhat_b = _fit_predict(_y_b, X_train.loc[_y_b.index], X_val)
rows_cmp.append({
    "condition": "(b) Differencing (lag=1)",
    **_eval(y_val, _yhat_b.cumsum() + y_train.iloc[-1]),
})

# (c) Detrending (linear polynomial, training only)
_t_dt = DetrendingTransform(degree=1, per_household=False)
_y_c  = _t_dt.fit_transform(y_train)
rows_cmp.append({
    "condition": "(c) Detrending (deg=1)",
    **_eval(y_val, _t_dt.inverse_transform(_fit_predict(_y_c, X_train, X_val))),
})

# (d) Deseasonalising (daily cycle, period=48)
_t_ds = DeseasonalisingTransform(period=48)
_y_d  = _t_ds.fit_transform(y_train)
rows_cmp.append({
    "condition": "(d) Deseasonalising",
    **_eval(y_val, _t_ds.inverse_transform(_fit_predict(_y_d, X_train, X_val))),
})

# (e) Deseason + Detrend — motivated by strong ACF at 48/336 plus weak trend
_y_e1  = _t_ds.transform(y_train)       # reuse fitted _t_ds
_t_dt2 = DetrendingTransform(degree=1, per_household=False).fit(_y_e1)
_y_e   = _t_dt2.transform(_y_e1)
rows_cmp.append({
    "condition": "(e) Deseason + Detrend",
    **_eval(
        y_val,
        _t_ds.inverse_transform(
            _t_dt2.inverse_transform(_fit_predict(_y_e, X_train, X_val))
        ),
    ),
})

# (f) Log + Deseason — motivated by White test heteroskedasticity
_t_log2 = LogTransform().fit(y_train)
_y_f1   = _t_log2.transform(y_train)
_t_ds2  = DeseasonalisingTransform(period=48).fit(_y_f1)
_y_f    = _t_ds2.transform(_y_f1)
rows_cmp.append({
    "condition": "(f) Log + Deseason",
    **_eval(
        y_val,
        _t_log2.inverse_transform(
            _t_ds2.inverse_transform(_fit_predict(_y_f, X_train, X_val))
        ),
    ),
})

comparison_df = pd.DataFrame(rows_cmp).set_index("condition")
_save_csv(comparison_df, "task04_transform_comparison.csv")
logger.info(
    "Transform comparison — %s  (val: %s → %s):\n%s",
    EVAL_HOUSEHOLD, VAL_START.date(), VAL_END.date(),
    comparison_df.to_string(),
)

# TODO: Plot the transformed series and their predictions to visually verify the results.
# TODO: Write which transforms performed best and whether the results align with the diagnostics.
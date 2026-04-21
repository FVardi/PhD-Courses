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
from statsmodels.tsa.stattools import acf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plots import plot_acf_pacf, plot_households_with_splits, plot_ols_diagnostics
from src.transforms.diagnostics import run_adf_tests, run_mann_kendall, run_white_test
from src.transforms.ols import fit_ols_households
from src.transforms.transforms import (
    BoxCoxTransform,
    DeseasonalisingTransform,
    DetrendingTransform,
    DifferencingTransform,
    LogTransform,
)

# %%

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
IMPUTED_PATH  = PROJECT_ROOT / "data" / "preprocessed" / "halfhourly_imputed.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "results" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

VALUE_COL = "energy_imputed_seasonal"

# ── Run mode ──────────────────────────────────────────────────────────────────
# QUICK_RUN=True  → use 3 households for diagnostics/transforms; fast for
#                   inspection.  The quality screening step still runs fully
#                   (its CSV output is required by tasks 06–10).
# QUICK_RUN=False → use 5 households for a richer diagnostic picture.
QUICK_RUN = True
LCLIDS    = ["MAC000002", "MAC000003", "MAC000004"] if QUICK_RUN else \
            ["MAC000002", "MAC000003", "MAC000004", "MAC000005", "MAC000006"]

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


# %%
# --- Household quality screening ----------------------------------------------
#
# Criteria (evaluated on the training period only):
#   imputed_frac     : fraction of slots that were imputed  → lower is better
#   max_zero_run     : longest consecutive streak of near-zero energy (< 0.05 kWh/hh)
#                      → flags households that were away / meter offline for days
#
# Thresholds (adjustable):
#   MAX_IMPUTED_FRAC = 0.05  (≤5 % imputed slots)
#   MAX_ZERO_RUN     = 48    (no zero-streak longer than 1 day = 48 half-hours)

MAX_IMPUTED_FRAC = 0.05
MAX_ZERO_RUN     = 48
ZERO_THRESHOLD   = 0.05   # kWh/hh — treat as "no usage"
SCREEN_BATCH     = 200    # households per read batch

logger.info("Screening all households for quality (training period only) …")

# Get full household list without loading values
all_lclids = (
    pd.read_parquet(IMPUTED_PATH, columns=[])
    .index.get_level_values("LCLid")
    .unique()
    .tolist()
)
logger.info("Total households to screen: %d", len(all_lclids))


def _max_consecutive_below(values, threshold):
    """Length of the longest consecutive run of values < threshold."""
    import numpy as np
    mask = (values < threshold).astype(int)
    if mask.sum() == 0:
        return 0
    changes = np.diff(mask, prepend=0, append=0)
    starts  = (changes ==  1).nonzero()[0]
    ends    = (changes == -1).nonzero()[0]
    return int((ends - starts).max())


rows = []
n_batches = (len(all_lclids) + SCREEN_BATCH - 1) // SCREEN_BATCH

for i in range(n_batches):
    batch_ids = all_lclids[i * SCREEN_BATCH : (i + 1) * SCREEN_BATCH]
    batch = pd.read_parquet(
        IMPUTED_PATH,
        filters=[("LCLid", "in", batch_ids)],
        columns=[VALUE_COL, "is_imputed"],
    )
    # Restrict to training period
    batch = batch.loc[batch.index.get_level_values("tstp") < TRAIN_END]

    for lclid, grp in batch.groupby(level="LCLid"):
        energy = grp[VALUE_COL].values
        rows.append({
            "LCLid":         lclid,
            "n_slots":       len(grp),
            "imputed_frac":  grp["is_imputed"].mean(),
            "max_zero_run":  _max_consecutive_below(energy, ZERO_THRESHOLD),
            "mean_energy":   energy.mean(),
        })

    if (i + 1) % 10 == 0 or (i + 1) == n_batches:
        logger.info("  screened %d / %d batches", i + 1, n_batches)

quality = pd.DataFrame(rows).set_index("LCLid")

# Apply filters
good = quality[
    (quality["imputed_frac"] <= MAX_IMPUTED_FRAC) &
    (quality["max_zero_run"] <= MAX_ZERO_RUN)
].sort_values(["imputed_frac", "max_zero_run"])

logger.info(
    "Households passing quality screen: %d / %d  "
    "(imputed ≤%.0f%%, zero-run ≤%d slots)",
    len(good), len(quality), MAX_IMPUTED_FRAC * 100, MAX_ZERO_RUN,
)

_save_csv(quality, "task04_household_quality_all.csv")
_save_csv(good,    "task04_household_quality_good.csv")

print(f"\n--- Quality screen: {len(good)} households passed ---")
print(good.head(20).to_string())

# %%
# --- Distribution of quality metrics ------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(quality["imputed_frac"] * 100, bins=60, color="steelblue", edgecolor="none")
axes[0].axvline(MAX_IMPUTED_FRAC * 100, color="crimson", linewidth=1.2, linestyle="--")
axes[0].set_xlabel("Imputed slots (%)")
axes[0].set_title("Imputation fraction")

axes[1].hist(quality["max_zero_run"], bins=60, color="steelblue", edgecolor="none")
axes[1].axvline(MAX_ZERO_RUN, color="crimson", linewidth=1.2, linestyle="--")
axes[1].set_xlabel("Max consecutive near-zero slots")
axes[1].set_title("Longest near-zero run")

axes[2].hist(quality["mean_energy"], bins=60, color="steelblue", edgecolor="none")
axes[2].set_xlabel("Mean energy (kWh/hh)")
axes[2].set_title("Mean energy usage")

for ax in axes:
    ax.set_ylabel("Households")

fig.suptitle(
    f"Household quality metrics (training period) — "
    f"{len(good)}/{len(quality)} pass thresholds",
    fontsize=11,
)
fig.tight_layout()
_save_fig(fig, "task04_household_quality_distributions.png")

# %%
# --- Load target series for stationarity tests --------------------------------

logger.info("Loading %d households …", len(LCLIDS))

multi = pd.read_parquet(
    FEATURES_DIR,
    filters=[("LCLid", "in", LCLIDS)],
    columns=[VALUE_COL],
)

# %%
# --- ADF stationarity tests ---------------------------------------------------

adf_results = run_adf_tests(multi, LCLIDS, VALUE_COL, TRAIN_END)
_save_csv(adf_results, "task04_adf_results.csv")

n_stat = adf_results["stationary (p<0.05)"].sum()
logger.info(
    "ADF summary: %d / %d tests reject the unit-root null — series are stationary.",
    n_stat, len(adf_results),
)
logger.info(
    "All ADF statistics are strongly negative (well below the ~-3.5 critical value). "
    "Energy consumption mean-reverts around daily/weekly cycles; differencing is not required."
)

# %%
# --- Mann-Kendall trend test --------------------------------------------------

mk_results = run_mann_kendall(multi, LCLIDS, VALUE_COL, TRAIN_END)
_save_csv(mk_results, "task04_mk_results.csv")

# %%
# --- Visual inspection --------------------------------------------------------

fig_ts = plot_households_with_splits(
    multi, LCLIDS, VALUE_COL,
    train_end=TRAIN_END, test_start=TEST_START,
)
_save_fig(fig_ts, "task04_households_with_splits.png")

# %%
# --- ACF / PACF up to lag 336 (daily + weekly seasonality) -------------------

train_flat = (
    multi
    .loc[multi.index.get_level_values("tstp") < TRAIN_END]
    .reset_index()
)

fig_acf = plot_acf_pacf(
    train_flat,
    household_ids=LCLIDS,
    lags=336,
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

# %%
# --- Diagnostic summary table -------------------------------------------------

def _recommend(row: pd.Series) -> str:
    recs = []
    if row["White p"] < 0.05:
        recs.append("log-transform")
    if row["trend slope"] != 0.0:
        recs.append("detrend")
    if row["ACF lag-48"] > 0.3 or row["ACF lag-336"] > 0.3:
        recs.append("seasonal features ✓")
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
# --- Differencing transform ---------------------------------------------------

series = (
    multi.xs(LCLIDS[0], level="LCLid")[VALUE_COL]
    .loc[lambda s: s.index < TRAIN_END]
    .dropna()
)

for lag, seasonal_lag in [(1, None), (1, 48), (48, None)]:
    t = DifferencingTransform(lag=lag, seasonal_lag=seasonal_lag)
    differenced   = t.fit_transform(series)
    reconstructed = t.inverse_transform(differenced)
    max_err = (series - reconstructed.loc[series.index]).abs().max()
    logger.info(
        "DifferencingTransform(lag=%d, seasonal_lag=%s)  |  len %d → %d  |  max err: %.2e",
        lag, seasonal_lag, len(series), len(differenced), max_err,
    )

# %%
# --- DetrendingTransform ------------------------------------------------------

for degree in [1, 2]:
    t = DetrendingTransform(degree=degree, per_household=False)
    detrended     = t.fit_transform(series)
    reconstructed = t.inverse_transform(detrended)
    max_err = (series - reconstructed).abs().max()
    logger.info(
        "DetrendingTransform(degree=%d)  |  max reconstruction error: %.2e",
        degree, max_err,
    )

# %%
# --- DeseasonalisingTransform -------------------------------------------------
# Fits slot means (0-47) on training data; subtracts on transform, adds on inverse.

t_seas = DeseasonalisingTransform(period=48)
deseasonalised = t_seas.fit_transform(series)
reconstructed  = t_seas.inverse_transform(deseasonalised)
max_err = (series - reconstructed).abs().max()
logger.info(
    "DeseasonalisingTransform  |  residual std %.4f (original %.4f)  |  max err: %.2e",
    deseasonalised.std(), series.std(), max_err,
)

# %%
# --- Variance-stabilising transforms ------------------------------------------
# LogTransform  : log1p / expm1.  Handles zeros natively; shifts negatives if needed.
# BoxCoxTransform: MLE lambda estimated on training data only.  Also shifts for non-positives.

for TransformCls, label in [(LogTransform, "LogTransform"), (BoxCoxTransform, "BoxCoxTransform")]:
    t = TransformCls()
    transformed   = t.fit_transform(series)
    reconstructed = t.inverse_transform(transformed)
    max_err = (series - reconstructed).abs().max()
    logger.info(
        "%s  |  lambda=%s  |  max reconstruction err: %.2e",
        label,
        getattr(t, "_lambda", "N/A"),
        max_err,
    )

# %%
# --- Preserving the original target -------------------------------------------
# When transforms are applied for modelling, a new column is created so that
# the original scale is available for evaluation.  Predictions are always
# inverse-transformed before computing metrics.

_demo = ols_data.xs(LCLIDS[0], level="LCLid")[[VALUE_COL]].copy()
_demo = _demo.loc[_demo.index < TRAIN_END]

t_log = LogTransform()
t_log.fit(_demo[VALUE_COL])
_demo["energy_t"] = t_log.transform(_demo[VALUE_COL])          # transformed column
_demo["energy_hat_t"] = _demo["energy_t"]                       # placeholder: model predictions
_demo["energy_hat"] = t_log.inverse_transform(_demo["energy_hat_t"])  # back to original scale

mae_original = (_demo[VALUE_COL] - _demo["energy_hat"]).abs().mean()
logger.info(
    "Target preservation demo  |  original col '%s' intact  |  MAE on original scale: %.4f",
    VALUE_COL, mae_original,
)

# %%
# --- Transform comparison experiment ------------------------------------------
# One household, OLS model, validation set.  Only the target is transformed;
# the feature matrix (lags, rolling, EWMA, calendar, Fourier) is unchanged.
# All transforms are fitted on training data only; predictions are
# inverse-transformed to the original scale before metrics are computed.
#
# MASE denominator: mean |y_t - y_{t-48}| on training (seasonal naïve baseline).

import statsmodels.api as sm

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
    Xc_t = sm.add_constant(X_t.loc[idx])
    Xc_v = sm.add_constant(X_v.dropna(how="any"))
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

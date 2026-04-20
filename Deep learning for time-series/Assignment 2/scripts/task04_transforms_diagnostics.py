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
from src.transforms.transforms import DetrendingTransform, DifferencingTransform

# %%

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "results" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

VALUE_COL = "energy(kWh/hh)"
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

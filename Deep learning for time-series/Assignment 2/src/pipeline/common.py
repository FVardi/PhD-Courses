"""
Shared constants and helpers used by the forecasting pipeline (Tasks 05–10).

Import with:
    from src.pipeline import VALUE_COL, make_feature_config, eval_metrics, ...
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder

from src.configs import FeatureConfig, MissingValueConfig, ModelConfig

logger = logging.getLogger(__name__)

# ── Target column ─────────────────────────────────────────────────────────────

VALUE_COL = "energy_imputed_seasonal"

# ── Time-split boundaries (used identically in every modelling script) ────────

TRAIN_END  = pd.Timestamp("2014-01-01")
VAL_START  = pd.Timestamp("2014-01-01")
VAL_END    = pd.Timestamp("2014-02-01")
TEST_START = pd.Timestamp("2014-02-01")
TEST_END   = pd.Timestamp("2014-03-01")

# ── Standard feature column groups ───────────────────────────────────────────
# All features are strictly causal (past-only, within-household).

LAG_COLS     = [f"lag_{l}" for l in [1, 2, 3, 4, 5, 6, 48, 336]]
ROLLING_COLS = [f"rolling_{s}_{w}" for s in ("mean", "std") for w in (6, 48, 336)]
EWMA_COLS    = [f"ewma_{s}" for s in (6, 48, 336)]
FOURIER_COLS = [
    f"fourier_{k}_{p}_{n}"
    for p in (48, 336) for n in (1, 2) for k in ("sin", "cos")
]

# ── LightGBM baseline hyperparameters (tuning deferred to Task 10) ────────────

LGBM_DEFAULTS = dict(
    num_leaves        = 31,
    n_estimators      = 500,
    learning_rate     = 0.05,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    random_state      = 42,
    verbose           = -1,
)


# ── Config factories ──────────────────────────────────────────────────────────

def make_feature_config(
    native_categorical: bool = False,
    extra_continuous:   list[str] | None = None,
    extra_categorical:  list[str] | None = None,
) -> FeatureConfig:
    """Return the standard FeatureConfig.

    Args:
        native_categorical : if True, include lclid_enc as a pd.Categorical
                             passthrough (native LightGBM handling).
        extra_continuous   : additional continuous columns (e.g. target-encoded
                             meta-features).
        extra_categorical  : additional OHE columns (e.g. OHE meta-features).
    """
    return FeatureConfig(
        timestamp_col               = "tstp",
        target_col                  = VALUE_COL,
        original_target_col         = VALUE_COL,
        continuous_features         = (
            LAG_COLS + ROLLING_COLS + EWMA_COLS + FOURIER_COLS
            + ["hour_of_day"] + (extra_continuous or [])
        ),
        categorical_features        = ["day_of_week", "month"] + (extra_categorical or []),
        boolean_features            = ["is_weekend"],
        native_categorical_features = ["lclid_enc"] if native_categorical else [],
        index_cols                  = ["LCLid", "tstp"],
    )


def make_missing_config() -> MissingValueConfig:
    """Return the standard MissingValueConfig."""
    return MissingValueConfig(
        ffill_cols     = LAG_COLS + ROLLING_COLS + EWMA_COLS,
        zero_fill_cols = FOURIER_COLS,
    )


def make_sklearn_model_config(estimator, model_name: str, normalize: bool = True) -> ModelConfig:
    """Wrap a sklearn estimator with standard OHE + StandardScaler flags."""
    return ModelConfig(
        estimator           = estimator,
        model_name          = model_name,
        normalize           = normalize,
        fill_missing        = True,
        encode_categoricals = True,
        categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    )


def make_lgbm_model_config(extra_params: dict | None = None) -> ModelConfig:
    """Return a LightGBM ModelConfig using LGBM_DEFAULTS (no scaling)."""
    params = {**LGBM_DEFAULTS, **(extra_params or {})}
    return ModelConfig(
        estimator           = LGBMRegressor(**params),
        model_name          = "LightGBM",
        normalize           = False,
        fill_missing        = True,
        encode_categoricals = True,
        categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def mase_denom(train_series: pd.Series) -> float:
    """Seasonal-naïve (lag-48) MAE on *train_series* — the MASE denominator."""
    return float(train_series.diff(48).abs().dropna().mean())


def eval_metrics(y_true: pd.Series, y_pred: pd.Series, denom: float) -> dict:
    """Return MAE / RMSE / MASE on the index intersection of y_true and y_pred."""
    idx  = y_true.index.intersection(y_pred.index)
    err  = y_true.loc[idx] - y_pred.loc[idx]
    mae  = float(err.abs().mean())
    rmse = float((err ** 2).mean() ** 0.5)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mase": round(mae / denom, 4)}


def per_hh_metrics(
    y_hat_val:  pd.Series,
    y_hat_test: pd.Series,
    val_pool:   pd.DataFrame,
    test_pool:  pd.DataFrame,
    train_pool: pd.DataFrame,
    cohort_ids: list[str],
    value_col:  str = VALUE_COL,
) -> pd.DataFrame:
    """Compute val and test MAE/RMSE/MASE for each household.

    MASE denominator is computed from each household's own training series —
    never from validation or test data.

    Returns a DataFrame indexed by LCLid with columns:
        val_mae, val_rmse, val_mase, test_mae, test_rmse, test_mase
    """
    records = []
    for lclid in cohort_ids:
        tr_m  = train_pool.index.get_level_values("LCLid") == lclid
        hv_m  = y_hat_val.index.get_level_values("LCLid")  == lclid
        ht_m  = y_hat_test.index.get_level_values("LCLid") == lclid
        if not tr_m.any() or not hv_m.any() or not ht_m.any():
            continue
        d = mase_denom(train_pool.loc[tr_m, value_col])
        if d == 0 or np.isnan(d):
            continue
        yv_true = val_pool.loc[val_pool.index.get_level_values("LCLid") == lclid, value_col]
        yt_true = test_pool.loc[test_pool.index.get_level_values("LCLid") == lclid, value_col]
        vm = eval_metrics(yv_true.reindex(y_hat_val.loc[hv_m].index),  y_hat_val.loc[hv_m],  d)
        tm = eval_metrics(yt_true.reindex(y_hat_test.loc[ht_m].index), y_hat_test.loc[ht_m], d)
        records.append({
            "LCLid":    lclid,
            "val_mae":  vm["mae"],  "val_rmse":  vm["rmse"],  "val_mase":  vm["mase"],
            "test_mae": tm["mae"],  "test_rmse": tm["rmse"],  "test_mase": tm["mase"],
        })
    return pd.DataFrame(records).set_index("LCLid")


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_cohort(artifacts_dir: Path, max_households: int) -> list[str]:
    """Return cohort LCLids from the Task 4 quality screen, sorted by quality.

    Raises FileNotFoundError if the quality CSV does not exist.
    """
    path = artifacts_dir / "task04_household_quality_good.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Quality CSV not found at {path}. Run task04 first."
        )
    quality = pd.read_csv(path, index_col="LCLid")
    cohort = (
        quality
        .sort_values(["imputed_frac", "max_zero_run"])
        .head(max_households)
        .index.tolist()
    )
    logger.info(
        "Cohort: %d / %d households (max_households=%d)",
        len(cohort), len(quality), max_households,
    )
    return cohort


def load_splits(
    features_dir: Path,
    cohort_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features for *cohort_ids* and return (train, val, test) splits.

    Splits respect the shared time boundaries; no future data leaks into train.
    """
    data = pd.read_parquet(features_dir, filters=[("LCLid", "in", cohort_ids)])
    tstp = data.index.get_level_values("tstp")
    train = data.loc[tstp < TRAIN_END].copy()
    val   = data.loc[(tstp >= VAL_START) & (tstp < VAL_END)].copy()
    test  = data.loc[(tstp >= TEST_START) & (tstp < TEST_END)].copy()
    logger.info(
        "Splits — train: %d  val: %d  test: %d  (households: %d)",
        len(train), len(val), len(test),
        data.index.get_level_values("LCLid").nunique(),
    )
    return train, val, test


def add_lclid_enc(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    cohort_ids: list[str],
) -> dict[str, int]:
    """Add *lclid_enc* integer column to each pool in-place.

    Returns the LCLid → integer mapping (derived from sorted cohort_ids so
    it is deterministic and independent of data ordering).
    """
    lclid_map = {lclid: i for i, lclid in enumerate(sorted(cohort_ids))}
    for pool in (train, val, test):
        pool["lclid_enc"] = pool.index.get_level_values("LCLid").map(lclid_map).values
    return lclid_map


# ── I/O helpers ──────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str, artifacts_dir: Path) -> None:
    path = artifacts_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", path)


def save_csv(df: pd.DataFrame, name: str, artifacts_dir: Path) -> None:
    path = artifacts_dir / name
    df.to_csv(path)
    logger.info("Saved table  → %s", path)

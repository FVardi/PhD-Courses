"""
Shared setup helper for tasks 13-15.

Reconstructs the best-variant encoding from Task 09 and the best
hyperparameters from Task 10, and returns fitted train/val/test pools
plus the feature and missing-value configs.

Imported by task13, task14, task15 — not intended to be run directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder

from src.configs import ModelConfig
from src.forecasting import MLForecast
from src.pipeline import (
    VALUE_COL,
    make_feature_config, make_missing_config,
    load_cohort, load_splits, add_lclid_enc,
)
from src.transforms.transforms import ComposedTransform, DeseasonalisingTransform, LogTransform

TARGET_TRANSFORM = ComposedTransform([LogTransform(), DeseasonalisingTransform(period=48)])


def load_tuned_setup(
    artifacts_dir: Path,
    features_dir: Path,
    acorn_path: Path,
    max_households: int = 50,
):
    """Return (cohort_ids, tr, va, te, feature_config, missing_config, best10_params).

    Loads the Task 09 best encoding variant and Task 10 best hyperparameters,
    reconstructs the encoded train/val/test pools, and returns everything
    needed to build and fit a wrapper.
    """
    with open(artifacts_dir / "task09_best_variant.json") as f:
        best9 = json.load(f)
    with open(artifacts_dir / "task10_best_params.json") as f:
        best10 = json.load(f)

    meta_cols = best9["meta_cols"]
    variant   = best9["variant"]

    cohort_ids = load_cohort(artifacts_dir, max_households)
    train_pool, val_pool, test_pool = load_splits(features_dir, cohort_ids)
    add_lclid_enc(train_pool, val_pool, test_pool, cohort_ids)

    meta = (
        pd.read_csv(acorn_path, usecols=["LCLid"] + meta_cols)
        .set_index("LCLid")
        .loc[lambda df: ~df.index.duplicated(keep="first")]
        .fillna("Unknown")
        .reindex(cohort_ids)
        .fillna("Unknown")
    )

    def _join(pool: pd.DataFrame) -> pd.DataFrame:
        pool = pool.copy()
        lclids = pool.index.get_level_values("LCLid")
        for col in meta_cols:
            pool[col] = lclids.map(meta[col]).values
        return pool

    if variant == "a_baseline":
        tr, va, te = train_pool, val_pool, test_pool
        extra_cont, extra_cat = [], []

    elif variant == "b_ohe":
        tr, va, te = _join(train_pool), _join(val_pool), _join(test_pool)
        extra_cont, extra_cat = [], meta_cols

    elif variant == "c_count":
        tr, va, te = _join(train_pool), _join(val_pool), _join(test_pool)
        for col in meta_cols:
            counts  = tr[col].value_counts()
            enc_col = f"{col}_cnt"
            for pool in (tr, va, te):
                pool[enc_col] = pool[col].map(counts).fillna(0).astype(float)
                pool.drop(columns=[col], inplace=True)
        extra_cont = [f"{c}_cnt" for c in meta_cols]
        extra_cat  = []

    else:  # d_target_enc
        tr, va, te = _join(train_pool), _join(val_pool), _join(test_pool)
        g_mean = train_pool[VALUE_COL].mean()
        for col in meta_cols:
            means   = tr.groupby(col)[VALUE_COL].mean()
            enc_col = f"{col}_te"
            for pool in (tr, va, te):
                pool[enc_col] = pool[col].map(means).fillna(g_mean).astype(float)
                pool.drop(columns=[col], inplace=True)
        extra_cont = [f"{c}_te" for c in meta_cols]
        extra_cat  = []

    feature_config = make_feature_config(
        native_categorical = True,
        extra_continuous   = extra_cont,
        extra_categorical  = extra_cat,
    )
    missing_config = make_missing_config()

    return cohort_ids, tr, va, te, feature_config, missing_config, best10["params"]


def build_wrapper(params: dict, feature_config, missing_config, target_transformer=TARGET_TRANSFORM) -> MLForecast:
    """Construct an MLForecast wrapper with LightGBM and the given params."""
    mc = ModelConfig(
        estimator           = LGBMRegressor(**{**params, "random_state": 42, "verbose": -1}),
        model_name          = "LightGBM",
        normalize           = False,
        fill_missing        = True,
        encode_categoricals = True,
        categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    )
    return MLForecast(mc, feature_config, missing_config, target_transformer)

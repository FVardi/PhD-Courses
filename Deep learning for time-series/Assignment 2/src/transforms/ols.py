"""
OLS baseline model for 1-step-ahead energy forecasting.

All engineered features are strictly causal (shifted/lagged), so regressing
energy(kWh/hh) on the feature matrix is a valid 1-step-ahead prediction
within any data split — no leakage.

Functions
---------
fit_ols_household   — Fit OLS for one household on a given split, return model + metrics
fit_ols_households  — Fit OLS for a list of households, return metrics summary DataFrame
"""

from __future__ import annotations

import logging

import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def fit_ols_household(
    data: pd.DataFrame,
    lclid: str,
    value_col: str,
    end: pd.Timestamp,
) -> tuple:
    """Fit OLS on training data for a single household.

    Args:
        data:      MultiIndex (LCLid, tstp) DataFrame with target + features.
        lclid:     Household ID.
        value_col: Target column name.
        end:       Exclusive upper bound of the split to fit on.

    Returns:
        (fitted_model, y, y_hat, residuals, metrics_dict)
    """
    train = (
        data.xs(lclid, level="LCLid")
        .loc[lambda df: df.index < end]
        .dropna()
    )

    drop_cols = {value_col} | {c for c in train.columns if c.startswith("energy") or c in ("gap_length", "is_imputed")}
    feature_cols = [c for c in train.columns if c not in drop_cols]
    X = sm.add_constant(train[feature_cols].astype(float))
    y = train[value_col].astype(float)

    model     = sm.OLS(y, X).fit()
    y_hat     = model.fittedvalues
    residuals = y - y_hat

    metrics = {
        "household": lclid,
        "R²":        round(model.rsquared, 4),
        "MAE":       round(residuals.abs().mean(), 4),
        "RMSE":      round((residuals ** 2).mean() ** 0.5, 4),
    }
    logger.info(
        "OLS %s  |  R²=%.4f  MAE=%.4f  RMSE=%.4f",
        lclid, metrics["R²"], metrics["MAE"], metrics["RMSE"],
    )
    return model, y, y_hat, residuals, metrics


def fit_ols_households(
    data: pd.DataFrame,
    lclids: list[str],
    value_col: str,
    end: pd.Timestamp,
) -> tuple[list[dict], list[tuple]]:
    """Fit OLS for each household and collect metrics.

    Returns:
        (metrics_list, results_list)
        where results_list contains (lclid, model, y, y_hat, residuals) per household.
    """
    metrics_list = []
    results_list = []
    for lclid in lclids:
        model, y, y_hat, residuals, metrics = fit_ols_household(
            data, lclid, value_col, end
        )
        metrics_list.append(metrics)
        results_list.append((lclid, model, y, y_hat, residuals))
    return metrics_list, results_list

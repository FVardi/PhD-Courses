"""
Stationarity and trend diagnostics for half-hourly energy time series.

Functions
---------
run_adf_tests       — Augmented Dickey-Fuller test across households and regression types
run_mann_kendall    — Mann-Kendall monotonic trend test across households
"""

from __future__ import annotations

import logging

import pandas as pd
import pymannkendall as mk
from statsmodels.stats.diagnostic import het_white
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def run_adf_tests(
    data: pd.DataFrame,
    lclids: list[str],
    value_col: str,
    train_end: pd.Timestamp,
    regressions: list[str] | None = None,
) -> pd.DataFrame:
    """Run ADF test for each household × regression combination on training data.

    H0: the series has a unit root (non-stationary).
    Reject H0 (p < 0.05) → evidence of stationarity.

    Args:
        data:        MultiIndex (LCLid, tstp) DataFrame containing value_col.
        lclids:      Household IDs to test.
        value_col:   Target column name.
        train_end:   Exclusive upper bound for training data.
        regressions: ADF regression types. Defaults to ["c", "ct", "ctt"].

    Returns:
        DataFrame indexed by (household, regression) with columns
        [ADF statistic, p-value, stationary (p<0.05)].
    """
    if regressions is None:
        regressions = ["c", "ct", "ctt"]

    rows = []
    for lclid in lclids:
        series = (
            data.xs(lclid, level="LCLid")[value_col]
            .loc[lambda s: s.index < train_end]
            .dropna()
        )
        for reg in regressions:
            stat, pval, *_ = adfuller(series, regression=reg, autolag="AIC")
            rows.append({
                "household":           lclid,
                "regression":          reg,
                "ADF statistic":       round(stat, 4),
                "p-value":             round(pval, 4),
                "stationary (p<0.05)": pval < 0.05,
            })
            logger.info("%s  reg=%-3s  stat=%.4f  p=%.4f", lclid, reg, stat, pval)

    return pd.DataFrame(rows).set_index(["household", "regression"])


def run_mann_kendall(
    data: pd.DataFrame,
    lclids: list[str],
    value_col: str,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    """Run Mann-Kendall trend test for each household on training data.

    H0: no monotonic trend.
    Reject H0 (p < 0.05) → a significant upward or downward trend is present.

    Args:
        data:      MultiIndex (LCLid, tstp) DataFrame containing value_col.
        lclids:    Household IDs to test.
        value_col: Target column name.
        train_end: Exclusive upper bound for training data.

    Returns:
        DataFrame indexed by household with columns
        [MK statistic, p-value, trend, no trend (p>=0.05)].
    """
    rows = []
    for lclid in lclids:
        series = (
            data.xs(lclid, level="LCLid")[value_col]
            .loc[lambda s: s.index < train_end]
            .dropna()
        )
        result = mk.original_test(series)
        rows.append({
            "household":         lclid,
            "MK statistic":      round(result.s, 4),
            "p-value":           round(result.p, 4),
            "trend":             result.trend,
            "slope":             round(result.slope, 6),
            "no trend (p≥0.05)": result.p >= 0.05,
        })
        logger.info("%s  s=%.4f  p=%.4f  trend=%s", lclid, result.s, result.p, result.trend)

    return pd.DataFrame(rows).set_index("household")


def run_white_test(
    results: list[tuple],
) -> pd.DataFrame:
    """Run White's heteroskedasticity test on OLS residuals for each household.

    H0: residuals are homoskedastic.
    Reject H0 (p < 0.05) → evidence of heteroskedasticity.

    Args:
        results: List of (lclid, model, y, y_hat, residuals) as returned by
                 fit_ols_households.

    Returns:
        DataFrame indexed by household with columns
        [LM statistic, LM p-value, heteroskedastic (p<0.05)].
    """
    rows = []
    for lclid, model, y, y_hat, residuals in results:
        lm, lm_pval, _, _ = het_white(residuals, model.model.exog)
        rows.append({
            "household":               lclid,
            "LM statistic":            round(lm, 4),
            "LM p-value":              round(lm_pval, 4),
            "heteroskedastic (p<0.05)": lm_pval < 0.05,
        })
        logger.info("%s  LM=%.4f  p=%.4f", lclid, lm, lm_pval)

    return pd.DataFrame(rows).set_index("household")

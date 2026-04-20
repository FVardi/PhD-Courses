"""
Invertible time-series transforms.

Each transform exposes:
    fit(series)               — learn parameters from data, return self
    transform(series)         — apply the transform
    fit_transform(series)     — fit then transform in one call
    inverse_transform(series) — reconstruct original scale/level

All state required for inversion is stored on the instance after fit().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseTransform(ABC):
    """Abstract base class for invertible time-series transforms."""

    @abstractmethod
    def fit(self, series: pd.Series) -> "BaseTransform":
        """Learn inversion state from *series*. Returns self."""

    @abstractmethod
    def transform(self, series: pd.Series) -> pd.Series:
        """Apply the transform. fit() must have been called first."""

    def fit_transform(self, series: pd.Series) -> pd.Series:
        """Convenience: fit then transform."""
        return self.fit(series).transform(series)

    @abstractmethod
    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """Reconstruct the original series from a transformed one."""

    def _check_fitted(self) -> None:
        if not self._is_fitted():
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling "
                "transform() or inverse_transform()."
            )

    def _is_fitted(self) -> bool:
        return getattr(self, "_fitted", False)


class DifferencingTransform(BaseTransform):
    """Differencing transform with optional seasonal differencing and exact inversion.

    Applies first-order differencing (lag), then optionally a second seasonal
    differencing pass (seasonal_lag).  Both steps are individually invertible;
    inversion is applied in reverse order.

    Inversion state
    ---------------
    _seed          : first *lag* values of the original series.
    _seasonal_seed : first *seasonal_lag* values of the intermediate
                     (once-differenced) series.  Only stored when
                     seasonal_lag is set.

    Parameters
    ----------
    lag : int
        First differencing period (1 → first difference).
    seasonal_lag : int or None
        Optional second differencing period applied after the first
        (48 → daily seasonal, 336 → weekly seasonal).

    Examples
    --------
    >>> t = DifferencingTransform(lag=1, seasonal_lag=48)
    >>> diff = t.fit_transform(series)
    >>> reconstructed = t.inverse_transform(diff)
    """

    def __init__(self, lag: int = 1, seasonal_lag: int | None = None) -> None:
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        if seasonal_lag is not None and seasonal_lag < 1:
            raise ValueError(f"seasonal_lag must be >= 1, got {seasonal_lag}")
        self.lag = lag
        self.seasonal_lag = seasonal_lag
        self._seed: pd.Series | None = None
        self._seasonal_seed: pd.Series | None = None
        self._fitted = False

    def fit(self, series: pd.Series) -> "DifferencingTransform":
        """Store seeds needed to invert both differencing passes."""
        self._seed = series.iloc[: self.lag].copy()
        if self.seasonal_lag is not None:
            intermediate = series.diff(self.lag).iloc[self.lag :]
            self._seasonal_seed = intermediate.iloc[: self.seasonal_lag].copy()
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        """Apply first-order then (optionally) seasonal differencing."""
        self._check_fitted()
        out = series.diff(self.lag).iloc[self.lag :]
        if self.seasonal_lag is not None:
            out = out.diff(self.seasonal_lag).iloc[self.seasonal_lag :]
        return out

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """Reconstruct original values.  Inversion order is seasonal → first."""
        self._check_fitted()
        out = series
        if self.seasonal_lag is not None:
            out = self._invert_single(out, self._seasonal_seed)
        out = self._invert_single(out, self._seed)
        return out

    # ------------------------------------------------------------------

    @staticmethod
    def _invert_single(series: pd.Series, seed: pd.Series) -> pd.Series:
        """Invert one differencing pass given its seed values."""
        lag = len(seed)
        out = np.empty(lag + len(series), dtype=float)
        out[:lag] = seed.values.astype(float)
        out[lag:] = series.values.astype(float)
        for i in range(lag, len(out)):
            out[i] += out[i - lag]
        if isinstance(seed.index, pd.DatetimeIndex) and isinstance(
            series.index, pd.DatetimeIndex
        ):
            idx = seed.index.append(series.index)
        else:
            idx = pd.RangeIndex(len(out))
        return pd.Series(out, index=idx, name=series.name)


class DetrendingTransform(BaseTransform):
    """Polynomial detrending transform with exact inversion.

    Fits a polynomial of *degree* 1 (linear) or 2 (quadratic) to the
    time series and subtracts it.  The inverse adds the stored trend back.

    The time axis is expressed in days from the first observed timestamp
    so the polynomial generalises correctly across train / val / test splits.

    Parameters
    ----------
    degree : int
        Polynomial degree.  Must be 1 or 2.
    per_household : bool
        If True and the input has a MultiIndex with a "LCLid" level, a
        separate polynomial is fitted per household.  If False, one global
        polynomial is fitted across all households.

    Inversion state
    ---------------
    _t0     : pd.Timestamp — origin of the time axis (days = 0).
    _coeffs : np.ndarray or dict[str, np.ndarray] — polynomial coefficients
              as returned by np.polyfit (highest degree first).

    Examples
    --------
    Per-household fit on a MultiIndex series:

    >>> t = DetrendingTransform(degree=1, per_household=True)
    >>> detrended = t.fit_transform(multi_series)
    >>> restored  = t.inverse_transform(detrended)

    Global fit on a plain Series:

    >>> t = DetrendingTransform(degree=1, per_household=False)
    >>> detrended = t.fit_transform(series)
    """

    def __init__(self, degree: int = 1, per_household: bool = True) -> None:
        if degree not in (1, 2):
            raise ValueError(f"degree must be 1 or 2, got {degree}")
        self.degree = degree
        self.per_household = per_household
        self._t0: pd.Timestamp | None = None
        self._coeffs: dict[str, np.ndarray] | np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, series: pd.Series) -> "DetrendingTransform":
        """Fit polynomial trend(s) to *series*."""
        self._t0 = self._get_timestamps(series).min()

        if self.per_household and isinstance(series.index, pd.MultiIndex):
            self._coeffs = {}
            for lclid, group in series.groupby(level="LCLid"):
                x = self._days(group)
                self._coeffs[lclid] = np.polyfit(x, group.values.astype(float), self.degree)
        else:
            x = self._days(series)
            self._coeffs = np.polyfit(x, series.values.astype(float), self.degree)

        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        """Subtract the fitted trend from *series*."""
        self._check_fitted()
        return (series - self._trend(series)).rename(series.name)

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """Add the fitted trend back to a detrended *series*."""
        self._check_fitted()
        return (series + self._trend(series)).rename(series.name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_timestamps(self, series: pd.Series) -> pd.DatetimeIndex:
        idx = series.index
        if isinstance(idx, pd.MultiIndex):
            return pd.DatetimeIndex(idx.get_level_values("tstp"))
        return pd.DatetimeIndex(idx)

    def _days(self, series: pd.Series) -> np.ndarray:
        """Numeric time axis in days since _t0."""
        ts = self._get_timestamps(series)
        return (ts - self._t0).total_seconds() / 86400.0

    def _trend(self, series: pd.Series) -> pd.Series:
        """Evaluate the fitted polynomial at each timestamp in *series*."""
        if isinstance(self._coeffs, dict):
            parts = []
            for lclid, group in series.groupby(level="LCLid"):
                x = self._days(group)
                vals = np.polyval(self._coeffs[lclid], x)
                parts.append(pd.Series(vals, index=group.index))
            return pd.concat(parts).reindex(series.index)
        else:
            x = self._days(series)
            return pd.Series(np.polyval(self._coeffs, x), index=series.index)

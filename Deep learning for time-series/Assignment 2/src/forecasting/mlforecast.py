"""
MLForecast — standardised forecasting wrapper.

Ensures every experiment uses identical rules for:
  - feature selection and X/y extraction  (FeatureConfig)
  - feature-matrix missing-value handling (MissingValueConfig)
  - categorical encoding and normalisation (ModelConfig)
  - target transformation and inversion    (target_transformer)
  - prediction alignment by timestamp      (original index preserved)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from src.configs.feature_config import FeatureConfig
from src.configs.missing_value_config import MissingValueConfig
from src.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class MLForecast:
    """Standardised forecasting wrapper for sklearn-compatible estimators.

    Parameters
    ----------
    model_config         : estimator + preprocessing flags.
    feature_config       : schema for feature extraction and target columns.
    missing_value_config : per-column fill rules for the feature matrix.
    target_transformer   : optional transform from src.transforms.transforms
                           (must implement fit_transform / inverse_transform).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        feature_config: FeatureConfig,
        missing_value_config: MissingValueConfig,
        target_transformer: Any = None,
    ) -> None:
        self.model_config         = model_config
        self.feature_config       = feature_config
        self.missing_value_config = missing_value_config
        self.target_transformer   = target_transformer

        self._estimator: Any              = None
        self._scaler: StandardScaler | None = None
        self._encoder: Any                = None
        self._encoded_cat_cols: list[str] = []
        self._native_cat_categories: dict = {}
        self._fitted: bool                = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MLForecast":
        """Fit the forecasting pipeline on training data.

        Steps (in order):
          1. Extract X, y, y_orig via FeatureConfig.
          2. Impute feature-matrix NaNs (if fill_missing).
          3. Encode categorical features (if encode_categoricals).
          4. Normalise continuous features (if normalize).
          5. Apply fit_transform on y (if target_transformer provided).
          6. Drop rows where X or y still contain NaN.
          7. Fit the cloned estimator.

        Args:
            df: Training DataFrame (MultiIndex or flat) containing both
                feature columns and target column(s).

        Returns:
            self
        """
        X, y, _ = self.feature_config.get_X_y(df)

        X = self._preprocess_X(X, fit=True)

        # Align y to X rows that survived preprocessing
        y = y.astype(float).reindex(X.index)

        if self.target_transformer is not None:
            y = self.target_transformer.fit_transform(y)

        # Drop any remaining NaN rows
        valid = X.notna().all(axis=1) & y.notna()
        X_clean = X.loc[valid]
        y_clean = y.loc[valid]

        self._estimator = self.model_config.clone_estimator()
        self._estimator.fit(X_clean, y_clean)
        self._fitted = True

        logger.info(
            "MLForecast.fit() — model=%s  rows=%d  features=%d",
            self.model_config.model_name, len(X_clean), X_clean.shape[1],
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate predictions on *df*, returned on the original target scale.

        Steps mirror fit() except target transformation is inverted at the end.
        Predictions are returned as a Series indexed identically to the
        valid (non-NaN) rows of the input.

        Args:
            df: DataFrame containing feature columns (target not required).

        Returns:
            pd.Series of predictions on the original scale, indexed by the
            input row index (timestamp, or MultiIndex household × timestamp).
        """
        if not self._fitted:
            raise RuntimeError(
                "MLForecast must be fitted before calling predict(). "
                "Call fit() first."
            )

        X = self.feature_config.get_X(df)
        X = self._preprocess_X(X, fit=False)

        valid   = X.notna().all(axis=1)
        X_clean = X.loc[valid]

        raw = self._estimator.predict(X_clean)
        y_hat = pd.Series(raw, index=X_clean.index, name=self.feature_config.target_col)

        if self.target_transformer is not None:
            y_hat = self.target_transformer.inverse_transform(y_hat)

        logger.info(
            "MLForecast.predict() — model=%s  rows=%d",
            self.model_config.model_name, len(y_hat),
        )
        return y_hat

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess_X(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Apply imputation → encoding → scaling to the feature matrix.

        Args:
            X   : raw feature matrix.
            fit : if True, fit internal transformers; otherwise transform only.

        Returns:
            Processed feature matrix (same index, possibly different columns
            after OHE expansion).
        """
        # 1. Missing-value imputation
        if self.model_config.fill_missing:
            X = self.missing_value_config.impute_missing_values(X)

        # 2. Categorical encoding (OneHotEncoder)
        cat_cols = [
            c for c in self.feature_config.categorical_features
            if c in X.columns
        ]
        if self.model_config.encode_categoricals and cat_cols:
            X_cat = X[cat_cols].astype(str)
            if fit:
                enc = clone(self.model_config.categorical_encoder)
                encoded = enc.fit_transform(X_cat)
                self._encoder = enc
                self._encoded_cat_cols = enc.get_feature_names_out(cat_cols).tolist()
            else:
                encoded = self._encoder.transform(X_cat)

            enc_df = pd.DataFrame(
                encoded,
                columns=self._encoded_cat_cols,
                index=X.index,
            )
            X = X.drop(columns=cat_cols).join(enc_df)

        # 3. Native categorical features → pd.Categorical (LightGBM passthrough)
        nat_cols = [
            c for c in self.feature_config.native_categorical_features
            if c in X.columns
        ]
        if nat_cols:
            if fit:
                self._native_cat_categories = {}
                for col in nat_cols:
                    cats = X[col].astype("category").cat.categories
                    self._native_cat_categories[col] = cats
                    X[col] = pd.Categorical(X[col], categories=cats)
            else:
                for col in nat_cols:
                    cats = self._native_cat_categories.get(col)
                    X[col] = pd.Categorical(X[col], categories=cats)

        # 4. Numeric scaling (StandardScaler on continuous + exogenous)
        scale_cols = [
            c for c in (
                self.feature_config.continuous_features
                + self.feature_config.exogenous_features
            )
            if c in X.columns
        ]
        if self.model_config.normalize and scale_cols:
            if fit:
                self._scaler = StandardScaler()
                X[scale_cols] = self._scaler.fit_transform(X[scale_cols])
            else:
                X[scale_cols] = self._scaler.transform(X[scale_cols])

        return X

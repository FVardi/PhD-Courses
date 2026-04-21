"""
FeatureConfig — schema definition and X/y extraction for the features DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FeatureConfig:
    """Defines the modelling schema of the prepared feature DataFrame.

    Parameters
    ----------
    timestamp_col        : name of the timestamp column / index level.
    target_col           : column used as modelling target (may be transformed).
    original_target_col  : column used for final evaluation (original scale).
    continuous_features  : numeric feature column names.
    categorical_features : categorical feature column names (will be OHE-encoded).
    boolean_features     : boolean / binary feature column names.
    index_cols           : index level names (e.g. ["LCLid", "tstp"]).
    exogenous_features          : optional additional numeric features.
    native_categorical_features : categorical features passed as pd.Categorical
                                  (not OHE-encoded; e.g. household_id for LightGBM).
    """

    timestamp_col: str
    target_col: str
    original_target_col: str
    continuous_features: list[str]
    categorical_features: list[str]
    boolean_features: list[str]
    index_cols: list[str]
    exogenous_features: list[str] = field(default_factory=list)
    native_categorical_features: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        all_features = (
            self.continuous_features
            + self.categorical_features
            + self.boolean_features
            + self.exogenous_features
            + self.native_categorical_features
        )

        if not all_features:
            raise ValueError("FeatureConfig must define at least one feature.")

        reserved = {self.target_col, self.original_target_col, self.timestamp_col}
        for col in reserved:
            if col in all_features:
                raise ValueError(
                    f"Column '{col}' must not appear in any feature list."
                )

        groups = {
            "continuous": set(self.continuous_features),
            "categorical": set(self.categorical_features),
            "boolean": set(self.boolean_features),
            "exogenous": set(self.exogenous_features),
            "native_categorical": set(self.native_categorical_features),
        }
        seen: set[str] = set()
        for gname, gcols in groups.items():
            overlap = seen & gcols
            if overlap:
                raise ValueError(
                    f"Columns {overlap} appear in more than one feature group "
                    f"(duplicate found in '{gname}')."
                )
            seen |= gcols

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def all_feature_cols(self) -> list[str]:
        return (
            self.continuous_features
            + self.categorical_features
            + self.boolean_features
            + self.exogenous_features
            + self.native_categorical_features
        )

    def get_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the feature matrix from *df*."""
        available = [c for c in self.all_feature_cols if c in df.columns]
        return df[available].copy()

    def get_X_y(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Return (X, y, y_orig) from *df*.

        X      : feature matrix
        y      : modelling target (``target_col``)
        y_orig : original-scale target (``original_target_col``)
        """
        X      = self.get_X(df)
        y      = df[self.target_col].copy()
        y_orig = df[self.original_target_col].copy()
        return X, y, y_orig

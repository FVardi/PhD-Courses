"""
ModelConfig — wraps an sklearn-compatible estimator with preprocessing flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.base import clone


@dataclass
class ModelConfig:
    """Defines how a forecasting model should be run.

    Parameters
    ----------
    estimator            : sklearn-compatible estimator (must implement fit/predict).
    model_name           : human-readable label used in logs and result tables.
    normalize            : apply StandardScaler to continuous features if True.
    fill_missing         : apply MissingValueConfig imputation if True.
    encode_categoricals  : apply the categorical encoder if True.
    categorical_encoder  : encoder instance (required when encode_categoricals=True).
    """

    estimator: Any
    model_name: str
    normalize: bool = True
    fill_missing: bool = True
    encode_categoricals: bool = False
    categorical_encoder: Any = field(default=None)

    def __post_init__(self) -> None:
        if self.encode_categoricals and self.categorical_encoder is None:
            raise ValueError(
                "categorical_encoder must be provided when encode_categoricals=True."
            )

    def clone_estimator(self) -> Any:
        """Return a fresh (unfitted) copy of the estimator."""
        return clone(self.estimator)

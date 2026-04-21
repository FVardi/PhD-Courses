"""
MissingValueConfig — feature-matrix imputation at model-initialisation time.

This is distinct from the raw-series imputation in Task 2.  It operates on
the engineered feature matrix just before model fitting / prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MissingValueConfig:
    """Specifies per-column fill strategies for the feature matrix.

    Fill order: backward-fill → forward-fill → zero-fill → fallback.

    Fallback (applied to any remaining NaNs):
        numeric columns   → column mean (computed from the data passed in)
        non-numeric cols  → string sentinel "missing"

    Parameters
    ----------
    bfill_cols     : columns to fill backward (from next valid observation).
    ffill_cols     : columns to fill forward (from previous valid observation).
    zero_fill_cols : columns to zero-fill.
    """

    bfill_cols: list[str] = field(default_factory=list)
    ffill_cols: list[str] = field(default_factory=list)
    zero_fill_cols: list[str] = field(default_factory=list)

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply configured fill rules then a sensible fallback.

        Args:
            df: Feature matrix (rows = observations, columns = features).

        Returns:
            Copy of *df* with NaNs filled according to the configured rules.
        """
        out = df.copy()

        for col in self.bfill_cols:
            if col in out.columns:
                out[col] = out[col].bfill()

        for col in self.ffill_cols:
            if col in out.columns:
                out[col] = out[col].ffill()

        for col in self.zero_fill_cols:
            if col in out.columns:
                out[col] = out[col].fillna(0)

        # Fallback: numeric → column mean; other → sentinel
        for col in out.columns:
            if out[col].isna().any():
                if pd.api.types.is_numeric_dtype(out[col]):
                    col_mean = out[col].mean()
                    out[col] = out[col].fillna(col_mean if pd.notna(col_mean) else 0.0)
                else:
                    out[col] = out[col].fillna("missing")

        return out

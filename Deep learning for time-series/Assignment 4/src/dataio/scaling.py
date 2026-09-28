"""Per-channel standardisation, fit on the training split only.

One (mean, sd) pair per channel, computed across every training series and timestep.
Deliberately NOT per-series: normalising each series individually would remove amplitude
differences between series, which are a legitimate discriminative signal. Note that the
archive has already applied per-series z-normalisation to FordA, FordB and AWR, so this
only does real work on Epilepsy.

Fit on the full training split rather than on the labelled subset: scaling reads no labels,
so it is not leakage, and it keeps the 100% and 10% regimes preprocessed identically.
"""

import json
from pathlib import Path

import numpy as np


def fit_scaler(X: np.ndarray) -> dict:
    """Return per-channel means and sds from X of shape (n_series, n_channels, length)."""
    if X.ndim != 3:
        raise ValueError(f"expected (n_series, n_channels, length), got shape {X.shape}")
    per_channel = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
    mean = np.nanmean(per_channel, axis=1)
    sd = np.nanstd(per_channel, axis=1)
    # A constant channel carries no information; dividing by its zero sd would produce NaN.
    sd = np.where(sd == 0, 1.0, sd)
    return {"mean": mean.tolist(), "sd": sd.tolist()}


def apply_scaler(X: np.ndarray, scaler: dict) -> np.ndarray:
    mean = np.asarray(scaler["mean"])[None, :, None]
    sd = np.asarray(scaler["sd"])[None, :, None]
    if mean.shape[1] != X.shape[1]:
        raise ValueError(f"scaler has {mean.shape[1]} channels, data has {X.shape[1]}")
    return (X - mean) / sd


def save_scaler(path: Path, scaler: dict) -> None:
    Path(path).write_text(json.dumps(scaler, indent=2), encoding="utf-8")


def load_scaler(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

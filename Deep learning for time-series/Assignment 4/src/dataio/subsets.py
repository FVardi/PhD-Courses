"""Stratified label subsets for the reduced-label regime.

The subset a seed produces is generated ONCE by dev/1_prepare_data.py and written to
results/label_subsets/. Every later part loads those indices rather than re-deriving them,
so a change here can never silently desynchronise Part B from Parts C-E - the saved file is
the shared contract that makes the comparisons paired.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_subset(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Return sorted indices of a stratified `fraction` of y."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return np.arange(len(y))

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
    indices, _ = next(splitter.split(np.zeros(len(y)), y))
    return np.sort(indices)


def subset_path(directory: Path, name: str, seed: int) -> Path:
    return Path(directory) / f"{name}_seed{seed}.json"


def save_indices(
    path: Path, indices: np.ndarray, *, dataset: str, seed: int, fraction: float,
    y: np.ndarray,
) -> None:
    """Write the indices plus enough context to audit them without the data to hand."""
    classes, counts = np.unique(y[indices], return_counts=True)
    payload = {
        "dataset": dataset,
        "seed": seed,
        "fraction": fraction,
        "n_selected": int(len(indices)),
        "n_available": int(len(y)),
        "per_class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
        "indices": [int(i) for i in indices],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_indices(path: Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(payload["indices"], dtype=np.int64)

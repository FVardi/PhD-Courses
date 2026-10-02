"""Read FD-A / FD-B from inside the TF-C checkout, for exploration only.

These two datasets never pass through src/dataio: Part F runs TF-C's own, unmodified code on
the files in its datasets/ folder (tfc_data in src/config.yaml). The notebooks read those
same files, so what is explored is exactly what TF-C consumes.

Each split is a dict {"samples": (n, 1, 5120), "labels": (n,)}. FD-A stores samples as a
numpy array and FD-B as a torch tensor; both come back here as float32 numpy in the
project-wide (n_series, n_channels, length) convention.
"""

from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
# From the TF-C README: 64 kHz recordings of rolling bearings, cut into 5120-step windows.
SAMPLING_HZ = 64_000
# The README names the three conditions (undamaged, inner damaged, outer damaged) but not
# which integer label is which, so the notebooks do not guess.
CLASS_NAMES = ["class 0", "class 1", "class 2"]


def load_split(cfg: dict, dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for `dataset` in {"fd_a", "fd_b"} and `split` in SPLITS."""
    import torch

    path = Path(cfg["tfc_data"][dataset]) / f"{split}.pt"
    # weights_only=False: FD-A pickles a numpy array, which the safe loader refuses.
    blob = torch.load(path, weights_only=False)
    X = np.asarray(blob["samples"], dtype=np.float32)
    y = np.asarray(blob["labels"]).astype(np.int64)
    if X.ndim != 3 or X.shape[0] != y.shape[0]:
        raise ValueError(f"{path}: unexpected shapes {X.shape} / {y.shape}")
    return X, y


def class_counts(cfg: dict, dataset: str) -> dict[str, np.ndarray]:
    """Labels-only pass over every split: {split: count per class}."""
    n_classes = len(CLASS_NAMES)
    return {s: np.bincount(load_split(cfg, dataset, s)[1], minlength=n_classes) for s in SPLITS}

"""Load the official train/test partitions, with a .npy cache.

The official partitions are used exactly as the archive ships them. Nothing here resamples
or re-splits; the only sampling decision in the pipeline is the stratified label subset in
subsets.py. The cache holds UNSCALED arrays - scaling is applied at use time from the saved
scaler, so data/raw stays the single source of truth.
"""

from pathlib import Path

import numpy as np

from src.dataio.ts import encode_labels, read_header, read_ts
from src.utils.config import ROOT

SPLITS = ("TRAIN", "TEST")


def raw_path(cfg: dict, name: str, split: str) -> Path:
    return ROOT / cfg["paths"]["data_raw"] / name / f"{name}_{split}.ts"


def processed_dir(cfg: dict) -> Path:
    path = ROOT / cfg["paths"]["data_processed"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def class_order(cfg: dict, name: str, split: str = "TRAIN") -> list[str]:
    """The dataset's class labels, in the order the .ts header declares them."""
    return read_header(raw_path(cfg, name, split))["class_order"]


def load_split(
    cfg: dict, name: str, split: str, use_cache: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return unscaled X of shape (n_series, n_channels, length) and integer labels y."""
    split = split.upper()
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    x_cache = processed_dir(cfg) / f"{name}_{split}_X.npy"
    y_cache = processed_dir(cfg) / f"{name}_{split}_y.npy"
    if use_cache and x_cache.exists() and y_cache.exists():
        return np.load(x_cache), np.load(y_cache)

    X, labels, header = read_ts(raw_path(cfg, name, split))
    y = encode_labels(labels, header["class_order"])

    expect = cfg["datasets"]["meta"].get(name)
    if expect is not None:
        n_expected = expect["n_train"] if split == "TRAIN" else expect["n_test"]
        actual = (len(X), X.shape[1], X.shape[2], len(header["class_order"]))
        wanted = (n_expected, expect["channels"], expect["length"], expect["classes"])
        if actual != wanted:
            raise ValueError(
                f"{name} {split}: got (n, channels, length, classes) {actual}, "
                f"config says {wanted} - run dev/0_fetch_data.py --verify-only"
            )

    np.save(x_cache, X)
    np.save(y_cache, y)
    return X, y

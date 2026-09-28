"""Shape, class balance and per-channel scale for one loaded dataset split.

summarise() is a pure function over arrays, so it is usable from a notebook or a REPL the
moment you have data in hand, without waiting on src/dataio. The __main__ path below needs
src/dataio to exist and data/raw to be populated.

Two config decisions marked TODO depend on what this prints:
  preprocessing.scaler  - per-channel vs global, visible in the per-channel scale spread
  corruption.replacement - a channel whose mean sits far from its range makes mean-fill a
                           conspicuous artefact rather than a neutral one
"""

import sys
from pathlib import Path

import numpy as np
import yaml

CONFIG = Path(__file__).resolve().parents[1] / "src" / "config.yaml"


def summarise(X: np.ndarray, y: np.ndarray, name: str = "", expect: dict | None = None) -> None:
    """Print a summary of one split. X is (n_series, n_channels, length), y is (n_series,).

    `expect` is an optional entry from datasets.meta in config.yaml; when given, the shape
    is checked against it so a wrong archive version is caught immediately.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (n_series, n_channels, length), got shape {X.shape}")
    n, c, t = X.shape
    print(f"=== {name or 'dataset'} ===")
    print(f"shape           {n} series x {c} channels x {t} steps")

    if expect is not None:
        for key, got in (("channels", c), ("length", t)):
            want = expect.get(key)
            if want is not None and want != got:
                print(f"  MISMATCH {key}: config says {want}, data has {got}")

    classes, counts = np.unique(y, return_counts=True)
    print(f"classes         {len(classes)}  (min {counts.min()} / max {counts.max()} per class)")
    if expect is not None and expect.get("classes") not in (None, len(classes)):
        print(f"  MISMATCH classes: config says {expect['classes']}, data has {len(classes)}")
    if counts.max() > 1.5 * counts.min():
        print("  imbalanced - macro-F1 and accuracy will diverge here")

    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan or n_inf:
        print(f"  {n_nan} NaN and {n_inf} inf values - imputation must be fit on train only")

    print("per-channel scale (over all series and timesteps):")
    print(f"  {'ch':>3} {'mean':>10} {'sd':>10} {'min':>10} {'max':>10}")
    sds = []
    for ch in range(c):
        v = X[:, ch, :]
        sd = float(np.nanstd(v))
        sds.append(sd)
        print(
            f"  {ch:>3} {float(np.nanmean(v)):>10.4f} {sd:>10.4f} "
            f"{float(np.nanmin(v)):>10.4f} {float(np.nanmax(v)):>10.4f}"
        )
        if sd == 0:
            print(f"      channel {ch} is constant - it carries no information")
    if c > 1 and min(sds) > 0 and max(sds) / min(sds) > 3:
        print(
            f"  channel sd spread is {max(sds) / min(sds):.1f}x - scale per channel, "
            "not globally"
        )
    print()


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    try:
        from src.dataio import load_split  # noqa: F401
    except ImportError:
        print(
            "src/dataio is not implemented yet, so there is nothing to load.\n"
            "Use summarise(X, y, name) directly on arrays you already have, or implement\n"
            "dev/0_fetch_data.py and dev/1_prepare_data.py first.",
            file=sys.stderr,
        )
        return 1

    for name in cfg["datasets"]["required"]:
        for split in ("train", "test"):
            X, y = load_split(name, split)
            summarise(X, y, f"{name} [{split}]", cfg["datasets"]["meta"].get(name))
    return 0


if __name__ == "__main__":
    sys.exit(main())

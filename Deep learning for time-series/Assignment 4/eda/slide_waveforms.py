"""Waveform figures for the presentation's datasets slide.

One PNG per dataset, each sized for a quarter of a 16:9 slide. A panel holds one class and
overlays a few training series of that class on one channel, so the figure shows whether
series of the same class line up in time - the property that decides whether a linear
probe on the raw signal can work.

Run from the assignment root:  python eda/slide_waveforms.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "eda")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from matplotlib import pyplot as plt

from plots import CATEGORICAL, INK, INK_MUTED, _style
from src.dataio.splits import class_order, load_split
from src.utils.config import load_config
import tfc_data

OUT = ROOT / "results" / "figures"
N_OVERLAY = 3
SEED = 42
BACKGROUND = "white"  # matches the slide, unlike the notebooks' SURFACE


def label(name: str) -> str:
    """'1.0' -> 'class 1', '-1' -> 'class -1', 'EPILEPSY' -> 'Epilepsy'."""
    try:
        return f"class {float(name):g}"
    except ValueError:
        return name if name.startswith("class") else name.capitalize()


def panel_figure(X, y, classes, names, channel, title, path, length=None):
    """One panel per class in `classes`; N_OVERLAY series of that class overlaid."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, len(classes), figsize=(5.6, 2.35), sharex=True, sharey=True)
    fig.patch.set_facecolor(BACKGROUND)
    for slot, (ax, cls) in enumerate(zip(np.atleast_1d(axes), classes)):
        idx = rng.choice(np.flatnonzero(y == cls), size=N_OVERLAY, replace=False)
        for i in idx:
            ax.plot(X[i, channel, :length], lw=0.9, alpha=0.85,
                    color=CATEGORICAL[slot % len(CATEGORICAL)])
        ax.set_title(label(names[cls]), loc="left", fontsize=9, color=INK, pad=4)
        ax.set_xlabel("timestep", color=INK_MUTED, fontsize=8)
        _style(ax)
        ax.set_facecolor(BACKGROUND)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=BACKGROUND)
    plt.close(fig)


def main() -> int:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)

    X, y = load_split(cfg, "FordA", "TRAIN")
    panel_figure(X, y, np.unique(y), class_order(cfg, "FordA"), 0,
                 "FordA - 3 series per class", OUT / "waveforms_forda.png")

    X, y = load_split(cfg, "ArticularyWordRecognition", "TRAIN")
    panel_figure(X, y, np.unique(y)[:2], class_order(cfg, "ArticularyWordRecognition"), 0,
                 "AWR - 3 series per class, 2 of 25 classes, channel 0",
                 OUT / "waveforms_awr.png")

    X, y = load_split(cfg, "Epilepsy", "TRAIN")
    classes = np.unique(y)
    panel_figure(X, y, classes[:3], class_order(cfg, "Epilepsy"), 0,
                 "Epilepsy - 3 series per class, 3 of 4 classes, channel 0",
                 OUT / "waveforms_epilepsy.png")

    X, y = tfc_data.load_split(cfg, "fd_b", "train")
    panel_figure(X, y, np.unique(y), tfc_data.CLASS_NAMES, 0,
                 "FD-B - 3 series per class, first 512 of 5,120 steps",
                 OUT / "waveforms_fdb.png", length=512)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Waveform figures for the presentation's datasets slide.

One PNG per dataset, each sized for a quarter of a 16:9 slide. Columns are classes and rows
are different training series of that class, one series per panel on one channel. Reading
down a column shows whether series of the same class line up in time - the property that
decides whether a linear probe on the raw signal can work. Overlaying the series in one
panel showed the same thing but was too cluttered at slide size.

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
N_ROWS = 2  # series shown per class, one per panel
SEED = 42
BACKGROUND = "white"  # matches the slide, unlike the notebooks' SURFACE


def label(name: str) -> str:
    """'1.0' -> 'class 1', '-1' -> 'class -1', 'EPILEPSY' -> 'Epilepsy'."""
    try:
        return f"class {float(name):g}"
    except ValueError:
        return name if name.startswith("class") else name.capitalize()


def panel_figure(X, y, classes, names, channel, title, path, length=None):
    """A grid with one column per class in `classes` and N_ROWS series of that class down
    the column, one series per panel. Axes are shared so amplitudes compare across panels."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(N_ROWS, len(classes), figsize=(5.6, 2.35), sharex=True,
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(BACKGROUND)
    for col, cls in enumerate(classes):
        idx = rng.choice(np.flatnonzero(y == cls), size=N_ROWS, replace=False)
        for row, i in enumerate(idx):
            ax = axes[row, col]
            ax.plot(X[i, channel, :length], lw=0.8, color=CATEGORICAL[col % len(CATEGORICAL)])
            _style(ax)
            ax.set_facecolor(BACKGROUND)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(label(names[cls]), loc="left", fontsize=9, color=INK, pad=3)
            if row == N_ROWS - 1:
                ax.set_xlabel("timestep", color=INK_MUTED, fontsize=8, labelpad=1)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    fig.savefig(path, dpi=200, facecolor=BACKGROUND)
    plt.close(fig)


def main() -> int:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)

    X, y = load_split(cfg, "FordA", "TRAIN")
    panel_figure(X, y, np.unique(y), class_order(cfg, "FordA"), 0,
                 "FordA - 2 series per class", OUT / "waveforms_forda.png")

    X, y = load_split(cfg, "ArticularyWordRecognition", "TRAIN")
    panel_figure(X, y, np.unique(y)[:2], class_order(cfg, "ArticularyWordRecognition"), 0,
                 "AWR - 2 series per class, 2 of 25 classes, channel 0",
                 OUT / "waveforms_awr.png")

    X, y = load_split(cfg, "Epilepsy", "TRAIN")
    classes = np.unique(y)
    panel_figure(X, y, classes[:3], class_order(cfg, "Epilepsy"), 0,
                 "Epilepsy - 2 series per class, 3 of 4 classes, channel 0",
                 OUT / "waveforms_epilepsy.png")

    X, y = tfc_data.load_split(cfg, "fd_b", "train")
    panel_figure(X, y, np.unique(y), tfc_data.CLASS_NAMES, 0,
                 "FD-B - 2 series per class, first 512 of 5,120 steps",
                 OUT / "waveforms_fdb.png", length=512)
    return 0


if __name__ == "__main__":
    sys.exit(main())

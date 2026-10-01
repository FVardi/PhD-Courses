"""Shared plotting for the dataset notebooks, so all four render identically.

Colour rules, chosen so colour never claims an identity it cannot carry:

  1 channel      one hue per class; the panel title names the class, so no legend
  2-3 channels   one categorical hue per channel, with a legend
  4+ channels    a single hue for every channel - colour carries NO channel identity
                 (a 9-step one-hue ramp fails the adjacent-lightness check, and cycling
                 categorical hues would be worse). Use plot_channels() to see channels
                 individually.
"""

import numpy as np
from matplotlib import pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e5e4e0"
# Validated categorical slots (all-pairs safe for the first three).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
SINGLE = "#2a78d6"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _grid_shape(n: int) -> tuple[int, int]:
    if n <= 2:
        return n, 1
    if n <= 4:
        return int(np.ceil(n / 2)), 2
    cols = min(5, int(np.ceil(np.sqrt(n))))
    return int(np.ceil(n / cols)), cols


def plot_class_examples(X, y, class_names, title, seed=42):
    """One example series per class, as small multiples on shared axes.

    X is (n_series, n_channels, length). One example per class is drawn with a fixed seed,
    so re-running shows the same series.
    """
    classes = np.unique(y)
    n_channels = X.shape[1]
    rows, cols = _grid_shape(len(classes))
    per_channel_colour = 2 <= n_channels <= 3
    per_class_colour = n_channels == 1 and len(classes) <= 3

    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols + 1.2, 1.9 * rows + 0.9), sharex=True, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    flat = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(seed)

    for slot, cls in enumerate(classes):
        ax = flat[slot]
        idx = int(rng.choice(np.flatnonzero(y == cls)))
        for ch in range(n_channels):
            if per_channel_colour:
                colour, alpha = CATEGORICAL[ch], 1.0
            elif per_class_colour:
                colour, alpha = CATEGORICAL[slot], 1.0
            else:
                colour, alpha = SINGLE, 0.75
            ax.plot(
                X[idx, ch, :], lw=1.0, color=colour, alpha=alpha, solid_capstyle="round",
                label=f"channel {ch}" if (per_channel_colour and slot == 0) else None,
            )
        ax.set_title(
            f"{class_names[slot]}    #{idx}", loc="left", fontsize=9.5, color=INK, pad=6
        )
        _style(ax)

    for ax in flat[len(classes):]:
        ax.set_visible(False)
    for ax in flat[len(classes) - cols if len(classes) > cols else 0:len(classes)]:
        ax.set_xlabel("timestep", color=INK_MUTED, fontsize=8.5)

    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)
    if per_channel_colour:
        handles, labels = flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper right", frameon=False, fontsize=8.5,
            labelcolor=INK_MUTED, ncol=len(labels),
        )
    fig.tight_layout()
    return fig


def plot_class_spectra(groups, class_names, sampling_hz, title):
    """Mean amplitude spectrum per class, one panel per class on shared log-y axes.

    `groups` maps a label to (X, y) with X as (n_series, 1, length); every group is drawn
    in every panel, so two datasets can be compared class by class. Each series has its own
    mean removed first: the 0 Hz bin would otherwise dwarf everything else on the axis.
    """
    classes = np.unique(np.concatenate([y for _, y in groups.values()]))
    fig, axes = plt.subplots(
        len(classes), 1, figsize=(8.4, 1.9 * len(classes) + 0.9), sharex=True, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    flat = np.atleast_1d(axes).ravel()

    for slot, cls in enumerate(classes):
        ax = flat[slot]
        for g, (label, (X, y)) in enumerate(groups.items()):
            series = X[y == cls, 0, :]
            if len(series) == 0:
                continue
            series = series - series.mean(axis=1, keepdims=True)
            amplitude = np.abs(np.fft.rfft(series, axis=1)).mean(axis=0) / series.shape[1]
            freqs = np.fft.rfftfreq(series.shape[1], d=1.0 / sampling_hz) / 1000.0
            ax.plot(freqs[1:], amplitude[1:], lw=1.0, color=CATEGORICAL[g],
                    label=label if slot == 0 else None)
        ax.set_yscale("log")
        ax.set_title(class_names[slot], loc="left", fontsize=9.5, color=INK, pad=6)
        _style(ax)

    flat[-1].set_xlabel("frequency (kHz)", color=INK_MUTED, fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)
    if len(groups) > 1:
        handles, labels = flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8.5,
                   labelcolor=INK_MUTED, ncol=len(labels))
    fig.tight_layout()
    return fig


def plot_channels(X, y, class_names, cls_index, title, seed=42):
    """Every channel of one example series, faceted - for datasets with too many channels
    to distinguish by colour."""
    classes = np.unique(y)
    cls = classes[cls_index]
    n_channels = X.shape[1]
    rows, cols = _grid_shape(n_channels)
    idx = int(np.random.default_rng(seed).choice(np.flatnonzero(y == cls)))

    fig, axes = plt.subplots(
        rows, cols, figsize=(3.0 * cols + 1.0, 1.7 * rows + 0.9), sharex=True, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    flat = np.atleast_1d(axes).ravel()
    for ch in range(n_channels):
        flat[ch].plot(X[idx, ch, :], lw=1.2, color=SINGLE, solid_capstyle="round")
        flat[ch].set_title(f"channel {ch}", loc="left", fontsize=9.5, color=INK, pad=6)
        _style(flat[ch])
    for ax in flat[n_channels:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{title} - {class_names[cls_index]}, series #{idx}",
        x=0.01, ha="left", fontsize=12.5, color=INK,
    )
    fig.tight_layout()
    return fig

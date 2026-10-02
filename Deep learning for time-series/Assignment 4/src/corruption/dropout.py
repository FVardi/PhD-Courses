"""Contiguous sensor dropout for Part D.

One unbroken interval per series is replaced by the training channel mean. The design was
fixed in src/config.yaml before any Part D result was inspected; the reasoning for each
choice is recorded there.

Applied to the raw signal, before the per-channel scaler and before the encoder, because it
models a sensor failing in the physical signal. Corrupting representations instead would
test nothing real - no sensor zeroes out a learned feature.

Because the fill is the training channel mean and scaling follows, the blanked region lands
at exactly 0 after scaling. For a linear probe those features then contribute nothing: the
model is starved of them rather than misled. That is deliberately the gentlest convention
for the raw baseline, so a surviving SSL advantage cannot be an artefact of the fill.
"""

import numpy as np


def blank_length(length: int, proportion: float) -> int:
    """Number of timesteps removed from a series of `length`."""
    if not 0 < proportion < 1:
        raise ValueError(f"proportion must be in (0, 1), got {proportion}")
    return int(round(proportion * length))


def intervals(length: int, total: int, n_intervals: int, rng) -> list[tuple[int, int]]:
    """`n_intervals` non-overlapping [start, stop) blocks covering `total` timesteps.

    The time axis is partitioned into n_intervals equal segments and one block is placed
    uniformly at random within each, which guarantees they cannot overlap. For the
    configured n_intervals=1 this reduces to a single start drawn uniformly over the series.
    """
    if n_intervals < 1:
        raise ValueError(f"n_intervals must be >= 1, got {n_intervals}")
    per_segment = length / n_intervals
    per_block = max(1, total // n_intervals)
    out = []
    for k in range(n_intervals):
        lo = int(round(k * per_segment))
        hi = int(round((k + 1) * per_segment))
        latest = max(lo, hi - per_block)
        start = int(rng.integers(lo, latest + 1))
        out.append((start, min(start + per_block, length)))
    return out


def contiguous_dropout(
    X: np.ndarray,
    fill: np.ndarray,
    proportion: float,
    seed: int,
    n_intervals: int = 1,
) -> np.ndarray:
    """Return a corrupted copy of X, shape (n_series, n_channels, length).

    `fill` holds one value per channel - the training channel means, so the caller supplies
    statistics fitted on TRAIN even when corrupting TEST.

    Every channel loses the same interval. Blanking a single channel would cost a 9-channel
    dataset a ninth of what it costs a univariate one, and the per-dataset results would no
    longer be comparable.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (n_series, n_channels, length), got shape {X.shape}")
    fill = np.asarray(fill, dtype=float)
    if fill.shape != (X.shape[1],):
        raise ValueError(f"fill must have one value per channel: {fill.shape} vs {X.shape[1]}")

    n, _, length = X.shape
    total = blank_length(length, proportion)
    rng = np.random.default_rng(seed)
    out = X.copy()

    for i in range(n):
        for start, stop in intervals(length, total, n_intervals, rng):
            out[i, :, start:stop] = fill[:, None]
    return out


def from_config(X: np.ndarray, scaler: dict, cfg: dict, seed: int,
                proportion: float | None = None,
                n_intervals: int | None = None) -> np.ndarray:
    """Apply the corruption exactly as src/config.yaml specifies it.

    `proportion` and `n_intervals` override the configured values for the Part E
    sensitivity sweep only. Everything else - fill, channel scope, start rule, seed - stays
    as configured, so the sweep's 20% x 1 cell produces the Part D arrays bit for bit.
    """
    spec = cfg["corruption"]
    if spec["replacement"] != "train_channel_mean":
        raise NotImplementedError(f"unsupported replacement: {spec['replacement']}")
    if spec["channels"] != "all":
        raise NotImplementedError(f"unsupported channel scope: {spec['channels']}")
    if spec["interval_start"] != "uniform":
        raise NotImplementedError(f"unsupported interval_start: {spec['interval_start']}")
    return contiguous_dropout(
        X,
        fill=np.asarray(scaler["mean"]),
        proportion=spec["proportion"] if proportion is None else proportion,
        seed=seed,
        n_intervals=spec["n_intervals"] if n_intervals is None else n_intervals,
    )

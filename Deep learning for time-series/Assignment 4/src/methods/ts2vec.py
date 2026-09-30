"""Adapter around the official TS2Vec checkout.

The checkout stays at its verified commit and is never modified. Unlike T-Loss it imports
cleanly on Python 3.13, so the only integration work here is shape and bookkeeping.

Shape. This project's convention is (n_series, n_channels, length); TS2Vec expects
(n_instance, n_timestamps, n_features). Every array crossing into the checkout is
transposed here, in one place, so no caller has to remember which way round it goes.

Training budget. TS2Vec's fit() picks n_iters itself when not told: 200 for a training
array of at most 100000 elements, 600 otherwise. default_iters() reproduces that rule
exactly so the value can be passed explicitly and recorded per run - the behaviour is
identical to letting fit() decide, but the number ends up in the run record instead of
being implicit.

Representations use encoding_window='full_series', which is what the checkout's own
tasks/classification.py uses for one-label-per-series problems. It max-pools over time to
give a single vector per series.
"""

import sys
from pathlib import Path

import numpy as np

from src.methods._checkout import activate, other_roots

ENCODING_WINDOW = "full_series"


def install_checkout(path: str | Path, others: list[str] | None = None) -> None:
    """Make the TS2Vec checkout the active one.

    `others` are the competing checkout paths; T-Loss ships a colliding top-level utils.py.
    See src/methods/_checkout.py.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"TS2Vec checkout not found at {root}")
    activate(root, others or [])


def to_checkout_layout(X: np.ndarray) -> np.ndarray:
    """(n_series, n_channels, length) -> (n_series, length, n_channels), float32."""
    if X.ndim != 3:
        raise ValueError(f"expected (n_series, n_channels, length), got shape {X.shape}")
    return np.ascontiguousarray(X.transpose(0, 2, 1), dtype=np.float32)


def default_iters(X: np.ndarray) -> int:
    """TS2Vec's own default rule, applied to the array as the checkout would see it."""
    return 200 if to_checkout_layout(X).size <= 100_000 else 600


def build(cfg: dict, input_dims: int, device: str = "cuda"):
    """Construct an untrained TS2Vec with the published defaults.

    Seed before calling this: the encoder's weights are initialised here.
    Only input_dims and device are set; everything else is left at the checkout's own
    defaults (output_dims=320, hidden_dims=64, depth=10, lr=0.001, batch_size=16).
    """
    install_checkout(cfg["checkouts"]["ts2vec"]["path"], other_roots(cfg, "ts2vec"))
    from ts2vec import TS2Vec

    return TS2Vec(input_dims=input_dims, device=device)


def pretrain(model, X: np.ndarray, n_iters: int | None = None, verbose: bool = False):
    """Train on X of shape (n_series, n_channels, length). Labels are never involved."""
    data = to_checkout_layout(X)
    model.fit(data, n_iters=n_iters if n_iters is not None else default_iters(X),
              verbose=verbose)
    return model


def encode(model, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """Return frozen representations of shape (n_series, output_dims).

    encode() switches the network to eval() and restores its previous mode afterwards, so
    extraction cannot alter the encoder.
    """
    return model.encode(
        to_checkout_layout(X), encoding_window=ENCODING_WINDOW, batch_size=batch_size
    )


def save_encoder(model, path: str | Path) -> Path:
    path = Path(path)
    model.save(str(path))
    return path


def load_encoder(model, path: str | Path):
    model.load(str(path))
    return model

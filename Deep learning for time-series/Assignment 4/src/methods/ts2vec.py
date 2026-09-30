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
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.methods._checkout import activate, other_roots

ENCODING_WINDOW = "full_series"
MASK_MODES = ("binomial", "continuous")


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


def build(cfg: dict, input_dims: int, device: str = "cuda", mask_mode: str = "binomial"):
    """Construct an untrained TS2Vec with the published defaults.

    Seed before calling this: the encoder's weights are initialised here.
    Only input_dims and device are set; everything else is left at the checkout's own
    defaults (output_dims=320, hidden_dims=64, depth=10, lr=0.001, batch_size=16).

    mask_mode (Part E). TSEncoder accepts it, but TS2Vec.__init__ never passes it on, so
    through the public class it is fixed at 'binomial'. It is set here after construction
    instead of editing the checkout. Two networks must be changed: _net, the one fit()
    trains, and net.module, the weight-averaged deep copy that encode() uses. The copy only
    runs in eval mode, where masking is off, but leaving it at the old value would make the
    saved object misreport how it was trained. The default leaves the model byte-identical
    to the Part C construction.
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
    install_checkout(cfg["checkouts"]["ts2vec"]["path"], other_roots(cfg, "ts2vec"))
    from ts2vec import TS2Vec

    model = TS2Vec(input_dims=input_dims, device=device)
    for net in (model._net, model.net.module):
        # Guard against a checkout where the attribute has moved: setattr would otherwise
        # create a dead attribute and the ablation would silently train binomial twice.
        if not hasattr(net, "mask_mode"):
            raise AttributeError("TSEncoder has no mask_mode; checkout differs from b0088e1")
        net.mask_mode = mask_mode
    return model


@contextmanager
def count_mask_calls(model):
    """Count calls to the checkout's two mask generators while the block runs.

    Yields a dict {'binomial': n, 'continuous': n}. TSEncoder.forward() looks the
    generators up as module globals at call time, so replacing them in the module that
    defines TSEncoder (found from the model itself, not by import name) is seen by every
    forward pass. This is how Part E shows which mask was used, from the calls themselves
    rather than from the attribute it set.
    """
    encoder = sys.modules[type(model._net).__module__]
    names = {"binomial": "generate_binomial_mask", "continuous": "generate_continuous_mask"}
    counts = dict.fromkeys(names, 0)
    originals = {mode: getattr(encoder, name) for mode, name in names.items()}

    def counted(mode):
        def wrapper(*args, **kwargs):
            counts[mode] += 1
            return originals[mode](*args, **kwargs)
        return wrapper

    for mode, name in names.items():
        setattr(encoder, name, counted(mode))
    try:
        yield counts
    finally:
        for mode, name in names.items():
            setattr(encoder, name, originals[mode])


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

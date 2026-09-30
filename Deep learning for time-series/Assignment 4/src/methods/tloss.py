"""Adapter around the official T-Loss checkout (UnsupervisedScalableRepresentationLearning).

The checkout stays at its verified commit and is never modified; everything needed to run
it from this repository lives here. Nothing in this file touches the learning objective -
the triplet loss, the causal CNN and the training loop are all the originals.

Two pieces of compatibility work are required.

1. Python 3.12 removed ``FileFinder.find_module``, which the checkout's ``losses/__init__``
   and ``networks/__init__`` both rely on to walk their own packages. install_checkout()
   builds those two packages directly and registers them in sys.modules, so the original
   __init__ never runs. The result is what those files intend: every submodule imported and
   exposed as an attribute.

2. The published hyperparameters live in default_hyperparameters.json rather than in the
   constructor defaults, which differ. We load the JSON and override only ``in_channels``
   (from the data) and ``cuda``/``gpu`` - the same three keys the checkout's own
   ucr.py:fit_hyperparameters overrides.

T-Loss takes no seed argument, so reproducibility comes from seeding before build(): the
encoder's weights are created in the constructor, and fit() draws its batches afterwards.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

from src.methods._checkout import activate, other_roots

ARCHITECTURE = "CausalCNN"
_PACKAGES = ("losses", "networks")


def install_checkout(path: str | Path, others: list[str] | None = None) -> None:
    """Make the T-Loss checkout the active one, with its packages pre-built.

    `others` are the competing checkout paths; both ship a top-level utils.py, so whichever
    was imported last must be evicted. See src/methods/_checkout.py.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"T-Loss checkout not found at {root}")
    activate(root, others or [])

    for name in _PACKAGES:
        if name in sys.modules:
            continue
        directory = root / name
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
        exported = []
        for file in sorted(directory.glob("*.py")):
            if file.stem == "__init__":
                continue
            qualified = f"{name}.{file.stem}"
            spec = importlib.util.spec_from_file_location(qualified, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
            setattr(package, file.stem, module)
            exported.append(file.stem)
        package.__all__ = exported


def hyperparameters(cfg: dict) -> dict:
    """The checkout's published defaults, read from its own default_hyperparameters.json."""
    path = Path(cfg["checkouts"]["tloss"]["path"]) / "default_hyperparameters.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build(cfg: dict, in_channels: int, cuda: bool = True, gpu: int = 0):
    """Construct an untrained encoder with the published hyperparameters.

    Seed before calling this: the encoder's weights are initialised here.
    """
    install_checkout(cfg["checkouts"]["tloss"]["path"], other_roots(cfg, "tloss"))
    import scikit_wrappers

    params = hyperparameters(cfg)
    params["in_channels"] = in_channels   # must match the data, not the JSON's default of 1
    params["cuda"] = cuda
    params["gpu"] = gpu

    model = scikit_wrappers.CausalCNNEncoderClassifier()
    model.set_params(**params)
    return model


def pretrain(model, X: np.ndarray, verbose: bool = False):
    """Train the encoder unsupervisedly on X of shape (n_series, n_channels, length).

    fit_encoder(), never fit(): fit() would additionally train the checkout's own internal
    SVM, and the probes in this project must be the ones from src/probes.
    y stays None, which disables the early-stopping heuristic - matching the published
    ``early_stopping: null``.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (n_series, n_channels, length), got shape {X.shape}")
    # The checkout calls encoder.double(); float64 input avoids a dtype mismatch.
    model.fit_encoder(np.ascontiguousarray(X, dtype=np.float64), y=None, verbose=verbose)
    return model


def encode(model, X: np.ndarray, batch_size: int = 50) -> np.ndarray:
    """Return frozen representations of shape (n_series, out_channels)."""
    return model.encode(np.ascontiguousarray(X, dtype=np.float64), batch_size=batch_size)


def save_encoder(model, prefix: str | Path) -> Path:
    """Write the encoder weights. Deliberately not model.save(), which also pickles the
    internal SVM through sklearn.externals.joblib - removed from modern scikit-learn."""
    model.save_encoder(str(prefix))
    return Path(f"{prefix}_{ARCHITECTURE}_encoder.pth")


def load_encoder(model, prefix: str | Path):
    model.load_encoder(str(prefix))
    return model

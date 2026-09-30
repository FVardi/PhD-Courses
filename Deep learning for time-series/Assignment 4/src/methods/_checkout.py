"""Keep the external checkouts from colliding with each other in one process.

T-Loss and TS2Vec both ship a top-level ``utils.py``. Python caches modules by name, so
whichever checkout is imported first wins: TS2Vec's ``from utils import take_per_row`` then
resolves to T-Loss's utils and raises ImportError. Each pretraining script only touches one
method, but anything that uses both - extraction, evaluation - hits this immediately.

activate() makes one checkout the current one: its directory goes first on sys.path, the
others are removed from it, and any module already imported from another checkout is
evicted from sys.modules so the next import re-resolves against the right directory.
"""

import os
import sys
from collections.abc import Iterable
from pathlib import Path


def activate(root: str | Path, others: Iterable[str | Path]) -> None:
    """Make `root` the active checkout, displacing any of `others` already loaded."""
    root_str = str(Path(root).resolve())
    other_strs = {str(Path(o).resolve()) for o in others} - {root_str}

    sys.path[:] = [p for p in sys.path if p not in other_strs]
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if origin is None:
            paths = getattr(module, "__path__", None)
            origin = next(iter(paths), None) if paths else None
        if not origin:
            continue
        try:
            resolved = str(Path(origin).resolve())
        except OSError:
            continue
        if any(resolved.startswith(other + os.sep) for other in other_strs):
            del sys.modules[name]


def other_roots(cfg: dict, keep: str) -> list[str]:
    """Every checkout path in the config except `keep`."""
    return [spec["path"] for name, spec in cfg["checkouts"].items()
            if name != keep and isinstance(spec, dict) and "path" in spec]

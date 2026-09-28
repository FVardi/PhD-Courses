"""One place to seed every source of randomness a run touches."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and - if it is installed - PyTorch.

    Torch is imported lazily: Parts B and D need no deep learning stack, and importing
    torch there would make the baselines depend on it for nothing.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

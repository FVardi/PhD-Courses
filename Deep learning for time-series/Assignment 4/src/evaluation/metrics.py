"""Accuracy and macro-F1, reported together and never collapsed into one number.

Both are retained even where they disagree: accuracy follows the majority classes, macro-F1
weights every class equally, and on a 25-class problem with a handful of labels the two can
tell genuinely different stories.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # zero_division=0: a class the probe never predicts scores 0 rather than raising.
        # With ~1 labelled example per class on AWR this is expected, not exceptional.
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation; sd is 0.0 for a single value."""
    array = np.asarray(values, dtype=float)
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return float(array.mean()), sd

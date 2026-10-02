"""Fit a probe once on clean TRAIN, score it on clean and on corrupted TEST.

Shared by Part D (dev/7_) and Part E (dev/8_) so the two parts evaluate identically. If
they diverged, a difference between Part E's binomial arm and Part D's TS2Vec row could
come from the evaluation code rather than from the encoder.

    clean TRAIN ──► probe fitted ──┬──► clean TEST      ──► accuracy, macro-F1
                                   └──► corrupted TEST  ──► accuracy, macro-F1  ──► Delta

The probe is fitted once, so the only difference between the two scores is the test array.
"""

import json
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import score
from src.probes.probes import build, tune
from src.utils.seeding import set_seed

FIELDS = [
    "n_train", "n_features",
    "accuracy_clean", "macro_f1_clean", "accuracy_corrupt", "macro_f1_corrupt",
    "delta_accuracy", "delta_macro_f1", "params", "folds", "fit_seconds",
]


def clean_vs_corrupt(kind: str, A: np.ndarray, y_train: np.ndarray,
                     B_clean: np.ndarray, B_corrupt: np.ndarray, y_test: np.ndarray,
                     cfg: dict, seed: int, standardise: bool) -> dict:
    """Tune and fit `kind` on (A, y_train), score both test matrices. Returns a row fragment.

    standardise: StandardScaler on the representation, fitted on clean TRAIN only, as in
    Part C. Used for encoder outputs, whose per-dimension scales differ by up to ~11x;
    raw features are already on the input scaler's scale and skip it.
    """
    return clean_vs_many(kind, A, y_train, B_clean, {None: B_corrupt}, y_test,
                         cfg, seed, standardise)[None]


def clean_vs_many(kind: str, A: np.ndarray, y_train: np.ndarray,
                  B_clean: np.ndarray, B_corrupts: dict, y_test: np.ndarray,
                  cfg: dict, seed: int, standardise: bool) -> dict:
    """As clean_vs_corrupt, but for several corrupted test matrices keyed by setting.

    The probe is tuned and fitted ONCE, so every setting is scored by the same classifier
    and differences between settings come from the test arrays alone. This is also what
    makes the Part E sweep cheap: one fit per encoder instead of one per setting.
    Returns {key: row fragment}; fit_seconds is the shared tune+fit time.
    """
    if standardise:
        rep_scaler = StandardScaler().fit(A)
        A = rep_scaler.transform(A)
        B_clean = rep_scaler.transform(B_clean)
        B_corrupts = {k: rep_scaler.transform(B) for k, B in B_corrupts.items()}

    set_seed(seed)
    t0 = time.time()
    params, info = tune(kind, A, y_train, cfg, seed)
    model = build(kind, params, seed).fit(A, y_train)
    elapsed = time.time() - t0
    clean = score(y_test, model.predict(B_clean))

    rows = {}
    for key, B in B_corrupts.items():
        dirty = score(y_test, model.predict(B))
        rows[key] = {
            "n_train": len(y_train), "n_features": A.shape[1],
            "accuracy_clean": clean["accuracy"],
            "macro_f1_clean": clean["macro_f1"],
            "accuracy_corrupt": dirty["accuracy"],
            "macro_f1_corrupt": dirty["macro_f1"],
            "delta_accuracy": dirty["accuracy"] - clean["accuracy"],
            "delta_macro_f1": dirty["macro_f1"] - clean["macro_f1"],
            "params": json.dumps(params), "folds": info["folds"],
            "fit_seconds": round(elapsed, 2),
        }
    return rows

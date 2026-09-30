"""Logistic regression and RBF-SVM probes, tuned on training data only.

Both probes are used unchanged in Parts B-E; only the representation handed to them differs.
The RBF-SVM is deliberately NOT a linear probe - it is T-Loss's and TS2Vec's own downstream
protocol - so the two are reported separately rather than averaged.

Fold rule. Stratified CV needs every class on both sides of every split, so a class with a
single example makes any validation scheme uninformative: that example is either never
trained on or never scored. Hence:

    min class count >= 2  ->  k = min(configured folds, min class count), tune normally
    min class count == 1  ->  untunable; fall back to the a priori defaults in config.yaml

This is not specific to k-fold. A single train/validation split fails identically, and
sklearn refuses to stratify it at all.

The fallback is a fixed value chosen before any result was seen, NOT the parameters tuned
under a larger label budget. Borrowing across budgets would let a 10%-label cell benefit
from a choice made with 100% of the labels; a fixed default keeps the budget honest at the
cost of being untuned, which `info["tuned"]` records per row. In practice this path is
reached only by ArticularyWordRecognition at 10% labels, where 27 examples spread over 25
classes leave most classes with one example.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

KINDS = ("logistic_regression", "rbf_svm")
SELECTION_METRIC = "accuracy"


def build(kind: str, params: dict, seed: int):
    """Instantiate an unfitted probe with the given hyperparameters."""
    if kind == "logistic_regression":
        return LogisticRegression(max_iter=5000, random_state=seed, **params)
    if kind == "rbf_svm":
        return SVC(kernel="rbf", random_state=seed, **params)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def grid(kind: str, cfg: dict) -> dict:
    """The hyperparameter grid for `kind`, read from config.yaml."""
    if kind == "logistic_regression":
        return {"C": list(cfg["probes"]["logistic_regression"]["C"])}
    if kind == "rbf_svm":
        spec = cfg["probes"]["rbf_svm"]
        return {"C": list(spec["C"]), "gamma": list(spec["gamma"])}
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def defaults(kind: str, cfg: dict) -> dict:
    """The a priori hyperparameters for cells that cannot be tuned at all."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return dict(cfg["probes"]["untunable_defaults"][kind])


def usable_folds(y: np.ndarray, configured: int) -> int | None:
    """Return the number of stratified folds this label set supports, or None if it cannot
    be tuned at all."""
    _, counts = np.unique(y, return_counts=True)
    smallest = int(counts.min())
    if smallest < 2:
        return None
    return max(2, min(configured, smallest))


def tune(kind: str, X: np.ndarray, y: np.ndarray, cfg: dict, seed: int) -> tuple[dict, dict]:
    """Select hyperparameters by stratified CV on the training data only.

    Returns (best_params, info). info records how the choice was made so the write-up can
    state it per cell rather than claiming one uniform protocol.
    """
    configured = cfg["probes"]["tuning"]["folds"]
    folds = usable_folds(y, configured)
    if folds is None:
        _, counts = np.unique(y, return_counts=True)
        return defaults(kind, cfg), {
            "tuned": False,
            "folds": None,
            "reason": f"smallest class has {counts.min()} example(s); "
                      "no train/validation split can both fit and score it; "
                      "a priori defaults used",
        }

    search = GridSearchCV(
        build(kind, {}, seed),
        grid(kind, cfg),
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        scoring=SELECTION_METRIC,
        n_jobs=-1,
    )
    search.fit(X, y)
    return dict(search.best_params_), {
        "tuned": True,
        "folds": folds,
        "reason": (
            "" if folds == configured
            else f"reduced from {configured}: smallest class limits the split"
        ),
    }

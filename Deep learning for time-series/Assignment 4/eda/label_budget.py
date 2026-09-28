"""Is the 10% label regime actually trainable? Arithmetic only - runs with no data on disk.

A stratified k-fold needs at least k examples of every class, so a small subset of a
many-class dataset can make the tuning protocol in src/config.yaml impossible before any
model is fit. This script checks every (dataset, label regime) pair against that constraint.

Class balance is assumed uniform here. The UCR/UEA versions of these datasets are balanced
or near-balanced, but once dev/1_prepare_data.py has run this should be recomputed from the
real training labels - see summarise() in eda/dataset_summary.py.
"""

import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "src" / "config.yaml"


def largest_remainder(total: int, weights: list[float]) -> list[int]:
    """Allocate `total` items across classes proportionally, distributing the remainder.

    This is how a stratified subsampler splits a budget that does not divide evenly.
    """
    exact = [total * w for w in weights]
    base = [int(e) for e in exact]
    remainder = total - sum(base)
    order = sorted(range(len(exact)), key=lambda i: exact[i] - base[i], reverse=True)
    for i in order[:remainder]:
        base[i] += 1
    return base


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    folds = cfg["probes"]["tuning"]["folds"]
    meta = cfg["datasets"]["meta"]
    names = cfg["datasets"]["required"] + cfg["datasets"].get("optional", [])

    print(f"Tuning protocol: {cfg['probes']['tuning']['method']}, {folds} folds")
    print("Assuming balanced classes.\n")
    print(f"{'dataset':<26} {'labels':>7} {'n':>6} {'classes':>8} {'min/class':>10}  status")
    print("-" * 78)

    problems = []
    for name in names:
        m = meta[name]
        n_train, n_classes = m["n_train"], m["classes"]
        weights = [1 / n_classes] * n_classes
        for regime in cfg["label_regimes"]:
            n_sub = int(round(regime * n_train))
            per_class = largest_remainder(n_sub, weights)
            lo = min(per_class)
            ok = lo >= folds
            status = "ok" if ok else f"CANNOT form {folds} stratified folds"
            if not ok:
                problems.append((name, regime, lo))
            print(
                f"{name:<26} {regime:>6.0%} {n_sub:>6} {n_classes:>8} {lo:>10}  {status}"
            )
        print()

    if problems:
        print("Blocked combinations:")
        for name, regime, lo in problems:
            print(f"  {name} @ {regime:.0%} labels: {lo} example(s) per class, need {folds}")
        print(
            "\nsklearn's StratifiedKFold raises ValueError here, it does not degrade "
            "gracefully.\nThe tuning protocol needs a documented fallback for these cells."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

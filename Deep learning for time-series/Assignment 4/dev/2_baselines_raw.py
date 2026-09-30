"""Part B: logistic regression and RBF-SVM on the vectorised raw signal, 100% and 10% labels, 3 seeds.

Per dataset this fits 2 probes x 4 training sets: the full labelled training split once (it
is seed-independent, so one run), and the three saved 10% subsets. Each fit is evaluated
exactly once on the official test set.

The pipeline is deliberately the same one Parts C-E use, with the only difference being
which representation is applied:

    load official split -> scale (per channel, train-fit) -> REPRESENTATION -> probe -> score

Here the representation is src.representations.raw.encode, i.e. flattening. In Part C that
single call becomes a frozen T-Loss or TS2Vec encoder and nothing else changes.

Hyperparameters are selected by stratified CV on training data only, with the fold count
adapted to the label budget. Where the budget leaves a class with one example the cell
cannot be tuned at all and falls back to the a priori defaults in config.yaml, fixed before
any result was seen. The output records `tuned` per row so the write-up can state this
rather than imply uniform tuning.

    python dev/2_baselines_raw.py
    python dev/2_baselines_raw.py --datasets Epilepsy --probes logistic_regression
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.dataio.scaling import apply_scaler, load_scaler  # noqa: E402
from src.dataio.splits import load_split, processed_dir  # noqa: E402
from src.dataio.subsets import load_indices, subset_path  # noqa: E402
from src.evaluation.metrics import mean_sd, score  # noqa: E402
from src.probes.probes import KINDS, build, tune  # noqa: E402
from src.representations.raw import encode  # noqa: E402
from src.utils.config import load_config, results_dir  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

FIELDS = [
    "dataset", "probe", "label_fraction", "seed", "n_train", "n_features",
    "accuracy", "macro_f1", "params", "tuned", "folds", "tuning_note", "fit_seconds",
]


def prepare(cfg: dict, name: str):
    """Return flattened, scaled train and test matrices plus labels."""
    X_train, y_train = load_split(cfg, name, "TRAIN")
    X_test, y_test = load_split(cfg, name, "TEST")
    scaler = load_scaler(processed_dir(cfg) / f"{name}_scaler.json")
    return (
        encode(apply_scaler(X_train, scaler)),
        y_train,
        encode(apply_scaler(X_test, scaler)),
        y_test,
    )


def main() -> int:
    cfg = load_config()
    known = cfg["datasets"]["required"] + cfg["datasets"].get("optional", [])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=known,
                        default=cfg["datasets"]["required"])
    parser.add_argument("--probes", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--out", default="baselines_raw.csv")
    args = parser.parse_args()

    subsets_dir = results_dir(cfg, "label_subsets")
    out_path = results_dir(cfg, "metrics") / args.out
    rows: list[dict] = []

    for name in args.datasets:
        A_train, y_train, A_test, y_test = prepare(cfg, name)
        print(f"\n{name}  train {A_train.shape}  test {A_test.shape}")

        for kind in args.probes:
            # 100% labels, per seed. The labelled set does not depend on the seed but the
            # CV fold shuffle does, and that is enough to change the selected hyperparameter
            # and hence the score. Running it once would report a standard deviation of zero
            # that is an artefact of the procedure rather than a property of the result.
            accs, f1s = [], []
            for seed in cfg["seeds"]:
                set_seed(seed)
                t0 = time.time()
                params, info = tune(kind, A_train, y_train, cfg, seed)
                model = build(kind, params, seed).fit(A_train, y_train)
                result = score(y_test, model.predict(A_test))
                elapsed = time.time() - t0
                accs.append(result["accuracy"])
                f1s.append(result["macro_f1"])
                rows.append({
                    "dataset": name, "probe": kind, "label_fraction": 1.0, "seed": seed,
                    "n_train": len(y_train), "n_features": A_train.shape[1],
                    **result, "params": json.dumps(params), "tuned": info["tuned"],
                    "folds": info["folds"], "tuning_note": info["reason"],
                    "fit_seconds": round(elapsed, 2),
                })
            a_mu, a_sd = mean_sd(accs)
            f_mu, f_sd = mean_sd(f1s)
            print(f"  {kind:<20} 100% x{len(accs)}    acc {a_mu:.4f}+/-{a_sd:.4f}  "
                  f"f1 {f_mu:.4f}+/-{f_sd:.4f}")

            # --- 10% labels: the three saved subsets ---
            for fraction in [f for f in cfg["label_regimes"] if f < 1.0]:
                accs, f1s, tuned_flags = [], [], []
                for seed in cfg["seeds"]:
                    set_seed(seed)
                    idx = load_indices(subset_path(subsets_dir, name, seed))
                    A, b = A_train[idx], y_train[idx]
                    t0 = time.time()
                    # Where the budget leaves a class with one example, tune() returns the
                    # a priori defaults rather than anything selected under a larger budget.
                    params, info = tune(kind, A, b, cfg, seed)
                    tuned_flags.append(info["tuned"])
                    model = build(kind, params, seed).fit(A, b)
                    result = score(y_test, model.predict(A_test))
                    elapsed = time.time() - t0
                    accs.append(result["accuracy"])
                    f1s.append(result["macro_f1"])
                    rows.append({
                        "dataset": name, "probe": kind, "label_fraction": fraction,
                        "seed": seed, "n_train": len(b), "n_features": A.shape[1],
                        **result, "params": json.dumps(params), "tuned": info["tuned"],
                        "folds": info["folds"], "tuning_note": info["reason"],
                        "fit_seconds": round(elapsed, 2),
                    })
                a_mu, a_sd = mean_sd(accs)
                f_mu, f_sd = mean_sd(f1s)
                flag = "" if all(tuned_flags) else "  (untuned: a priori defaults)"
                print(f"  {kind:<20} {fraction:.0%} x{len(cfg['seeds'])}     "
                      f"acc {a_mu:.4f}+/-{a_sd:.4f}  f1 {f_mu:.4f}+/-{f_sd:.4f}{flag}")

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} rows -> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

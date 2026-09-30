"""Part C: fit the probes on frozen representations and evaluate once on the test set.

Deliberately the same procedure as dev/2_baselines_raw.py - same probes, same grids, same
fold rule, same seeds, same saved label subsets, one evaluation on the official test set.
The only difference is where the feature matrix comes from: cached encoder output here,
flattened raw signal there. That is what makes Part B and Part C a paired comparison.

This script never loads an encoder. It reads .npy files written by dev/5_, so it is
structurally incapable of updating encoder weights - frozen is enforced by the shape of the
pipeline rather than asserted.

Representation scaling. The 320 encoder dimensions have unconstrained scale (measured spread
7.8x for T-Loss, 10.8x for TS2Vec), so a StandardScaler is fitted on the TRAIN
representations and applied to both splits - the same fit-on-train rule as the input scaler
and, like it, reading no labels. Part B needs no equivalent step: per-channel scaling already
left its flattened columns within ~2x of unit variance, and adding one moves its numbers by
at most 0.0145. This does differ from TS2Vec's own protocol, which scales before its logistic
regression but not before its SVM; scaling uniformly keeps the probes comparable across
methods, and should if anything help TS2Vec rather than handicap it.

Writes results/metrics/ssl_probe.csv, one row per
(method, dataset, probe, label regime, seed).

    python dev/6_probe_eval.py
    python dev/6_probe_eval.py --methods ts2vec
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
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.dataio.splits import load_split  # noqa: E402
from src.dataio.subsets import load_indices, subset_path  # noqa: E402
from src.evaluation.metrics import mean_sd, score  # noqa: E402
from src.probes.probes import KINDS, build, tune  # noqa: E402
from src.utils.config import load_config, results_dir  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

METHODS = ("tloss", "ts2vec")
FIELDS = [
    "method", "dataset", "probe", "label_fraction", "seed", "n_train", "n_features",
    "accuracy", "macro_f1", "params", "tuned", "folds", "tuning_note", "fit_seconds",
]


def representations(out_dir: Path, method: str, dataset: str, seed: int):
    """Load the cached (train, test) representations, or None if extraction has not run."""
    paths = {s: out_dir / f"{method}_{dataset}_seed{seed}_{s}.npy" for s in ("TRAIN", "TEST")}
    if not all(p.exists() for p in paths.values()):
        return None
    return np.load(paths["TRAIN"]), np.load(paths["TEST"])


def append(path: Path, row: dict) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    cfg = load_config()
    known = cfg["datasets"]["required"] + cfg["datasets"].get("optional", [])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=known,
                        default=cfg["datasets"]["required"])
    parser.add_argument("--probes", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--out", default="ssl_probe.csv")
    args = parser.parse_args()

    reps_dir = results_dir(cfg, "representations")
    subsets_dir = results_dir(cfg, "label_subsets")
    out_path = results_dir(cfg, "metrics") / args.out
    # Drop only the rows this invocation will regenerate, so a partial run
    # (--methods / --datasets) cannot discard another method's results.
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            keep = [r for r in csv.DictReader(fh)
                    if not (r["method"] in args.methods and r["dataset"] in args.datasets
                            and r["probe"] in args.probes)]
        out_path.unlink()
        for row in keep:
            append(out_path, row)
    reduced = [f for f in cfg["label_regimes"] if f < 1.0]
    n_rows = skipped = 0

    for method in args.methods:
        for dataset in args.datasets:
            _, y_train = load_split(cfg, dataset, "TRAIN")
            _, y_test = load_split(cfg, dataset, "TEST")
            print(f"\n{method} / {dataset}")

            for kind in args.probes:
                # 100% labels. Unlike Part B this runs per seed: the labelled set is the
                # same every time, but the encoder that produced the features is not.
                accs, f1s = [], []
                for seed in cfg["seeds"]:
                    reps = representations(reps_dir, method, dataset, seed)
                    if reps is None:
                        skipped += 1
                        continue
                    A_train, A_test = reps
                    scaler = StandardScaler().fit(A_train)
                    A, B = scaler.transform(A_train), scaler.transform(A_test)

                    set_seed(seed)
                    t0 = time.time()
                    params, info = tune(kind, A, y_train, cfg, seed)
                    model = build(kind, params, seed).fit(A, y_train)
                    result = score(y_test, model.predict(B))
                    elapsed = time.time() - t0
                    accs.append(result["accuracy"])
                    f1s.append(result["macro_f1"])
                    append(out_path, {
                        "method": method, "dataset": dataset, "probe": kind,
                        "label_fraction": 1.0, "seed": seed, "n_train": len(y_train),
                        "n_features": A.shape[1], **result, "params": json.dumps(params),
                        "tuned": info["tuned"], "folds": info["folds"],
                        "tuning_note": info["reason"], "fit_seconds": round(elapsed, 2),
                    })
                    n_rows += 1
                if accs:
                    a_mu, a_sd = mean_sd(accs)
                    f_mu, f_sd = mean_sd(f1s)
                    print(f"  {kind:<20} 100% x{len(accs)}    acc {a_mu:.4f}+/-{a_sd:.4f}  "
                          f"f1 {f_mu:.4f}+/-{f_sd:.4f}")

                # Reduced label regimes, using the subsets Part B used.
                for fraction in reduced:
                    accs, f1s, note = [], [], ""
                    for seed in cfg["seeds"]:
                        reps = representations(reps_dir, method, dataset, seed)
                        if reps is None:
                            continue
                        A_train, A_test = reps
                        # Fitted on the FULL train split, not the labelled subset: scaling
                        # reads no labels, and a 320-dim scaler from 13 rows would be noise.
                        scaler = StandardScaler().fit(A_train)
                        A_all, B = scaler.transform(A_train), scaler.transform(A_test)

                        idx = load_indices(subset_path(subsets_dir, dataset, seed))
                        A, b = A_all[idx], y_train[idx]

                        set_seed(seed)
                        t0 = time.time()
                        # Where a class has a single example, tune() returns the a priori
                        # defaults from config.yaml - the same fallback dev/2_ uses, so the
                        # untunable cells are handled identically in Parts B and C.
                        params, info = tune(kind, A, b, cfg, seed)
                        model = build(kind, params, seed).fit(A, b)
                        result = score(y_test, model.predict(B))
                        elapsed = time.time() - t0
                        accs.append(result["accuracy"])
                        f1s.append(result["macro_f1"])
                        note = "" if info["tuned"] else "  (untuned: a priori defaults)"
                        append(out_path, {
                            "method": method, "dataset": dataset, "probe": kind,
                            "label_fraction": fraction, "seed": seed, "n_train": len(b),
                            "n_features": A.shape[1], **result, "params": json.dumps(params),
                            "tuned": info["tuned"], "folds": info["folds"],
                            "tuning_note": info["reason"], "fit_seconds": round(elapsed, 2),
                        })
                        n_rows += 1
                    if accs:
                        a_mu, a_sd = mean_sd(accs)
                        f_mu, f_sd = mean_sd(f1s)
                        print(f"  {kind:<20} {fraction:.0%} x{len(accs)}     "
                              f"acc {a_mu:.4f}+/-{a_sd:.4f}  f1 {f_mu:.4f}+/-{f_sd:.4f}{note}")

    print(f"\n{n_rows} rows -> {out_path.relative_to(ROOT)}")
    if skipped:
        print(f"{skipped} (method, dataset, seed) skipped - run dev/5_ once pretraining finishes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Load the official train/test partitions; fit the scaler; save the stratified label subsets.

Writes three things and transforms nothing:

  data/processed/{dataset}_{SPLIT}_X.npy    unscaled array cache, so the pure-Python .ts
  data/processed/{dataset}_{SPLIT}_y.npy    parser runs once rather than on every script
  data/processed/{dataset}_scaler.json      per-channel mean/sd fit on TRAIN only
  results/label_subsets/{dataset}_seed{seed}.json

The scaler is saved as parameters, not baked into the arrays: data/raw stays the single
source of truth, the numbers are small enough to read and defend in the write-up, and Parts
C-F may need the unscaled signal (a method that normalises its own input, or Part D's
train_channel_mean fill, which means something different once channel means are 0).

    python dev/1_prepare_data.py
    python dev/1_prepare_data.py --datasets Epilepsy --rebuild
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.dataio.scaling import fit_scaler, save_scaler  # noqa: E402
from src.dataio.splits import class_order, load_split, processed_dir  # noqa: E402
from src.dataio.subsets import save_indices, stratified_subset, subset_path  # noqa: E402
from src.utils.config import load_config, results_dir  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402


def main() -> int:
    cfg = load_config()
    known = cfg["datasets"]["required"] + cfg["datasets"].get("optional", [])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=known, default=known)
    parser.add_argument("--rebuild", action="store_true", help="ignore the .npy cache")
    args = parser.parse_args()

    subsets_dir = results_dir(cfg, "label_subsets")
    reduced = [f for f in cfg["label_regimes"] if f < 1.0]
    warnings: list[str] = []

    for name in args.datasets:
        X_train, y_train = load_split(cfg, name, "TRAIN", use_cache=not args.rebuild)
        X_test, y_test = load_split(cfg, name, "TEST", use_cache=not args.rebuild)
        names = class_order(cfg, name)

        print(f"{name}")
        print(
            f"  train {X_train.shape}  test {X_test.shape}  "
            f"{len(names)} classes  labels {names[0]!r}..{names[-1]!r}"
        )

        scaler = fit_scaler(X_train)
        scaler_file = processed_dir(cfg) / f"{name}_scaler.json"
        save_scaler(scaler_file, scaler)
        span = max(scaler["sd"]) / min(scaler["sd"])
        print(
            f"  scaler  {len(scaler['mean'])} channel(s), "
            f"sd {min(scaler['sd']):.3f}-{max(scaler['sd']):.3f} ({span:.1f}x spread)"
            f" -> {scaler_file.relative_to(ROOT)}"
        )

        for fraction in reduced:
            for seed in cfg["seeds"]:
                set_seed(seed)
                indices = stratified_subset(y_train, fraction, seed)
                path = subset_path(subsets_dir, name, seed)
                save_indices(
                    path, indices, dataset=name, seed=seed, fraction=fraction, y=y_train
                )
                _, counts = np.unique(y_train[indices], return_counts=True)
                note = ""
                folds = cfg["probes"]["tuning"]["folds"]
                if counts.min() < folds:
                    note = f"  <-- min {counts.min()}/class, {folds}-fold CV impossible"
                    warnings.append(f"{name} seed {seed}: {counts.min()} per class")
                print(
                    f"  {fraction:.0%} seed {seed}: {len(indices):>4} series, "
                    f"{counts.min()}-{counts.max()} per class{note}"
                )
        print()

    if warnings:
        print(f"{len(warnings)} (dataset, seed) subsets cannot support "
              f"{cfg['probes']['tuning']['folds']}-fold stratified CV.")
        print("src/probes needs a documented fallback before Part B can run on these.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

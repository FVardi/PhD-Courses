"""Part D: clean versus contiguous-dropout test performance, 100%-label models.

Nothing is retrained and nothing is re-pretrained. For each (representation, dataset, probe,
seed) the probe is fitted once on CLEAN training data and then scored twice - on the clean
official test set and on a corrupted copy of it. The only thing that differs between the two
columns is the test array, which is what lets the difference be attributed to robustness.

    clean TRAIN ──► probe fitted ──┬──► clean TEST      ──► accuracy, macro-F1
                                   └──► corrupted TEST  ──► accuracy, macro-F1  ──► Delta

Corruption enters before the scaler and before the encoder, because it models a sensor
failing in the physical signal: see src/corruption/dropout.py. Every representation sees the
identical corrupted arrays, since the corruption RNG is seeded by the same dataset seed.

Both columns are computed in this one run rather than copying the clean numbers across from
Parts B and C, so the clean column doubles as a consistency check against them.

Test-only corruption measures robustness to UNSEEN degradation, not learned invariance. A
method could degrade here and still be fine if trained with augmentation; the write-up should
say "clean-trained representations degrade less", not "method X is robust to dropout".

Writes results/metrics/corruption.csv.

    python dev/7_corruption_eval.py
    python dev/7_corruption_eval.py --representations raw ts2vec
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.corruption.dropout import from_config as corrupt  # noqa: E402
from src.dataio.scaling import apply_scaler, load_scaler  # noqa: E402
from src.dataio.splits import load_split, processed_dir  # noqa: E402
from src.evaluation.metrics import mean_sd  # noqa: E402
from src.evaluation.robustness import FIELDS as SCORE_FIELDS, clean_vs_corrupt  # noqa: E402
from src.methods import METHODS, encode as encode_frozen, load_pretrained  # noqa: E402
from src.probes.probes import KINDS  # noqa: E402
from src.representations.raw import encode as flatten  # noqa: E402
from src.utils.config import load_config, results_dir  # noqa: E402

warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)

REPRESENTATIONS = ("raw", *METHODS)
FIELDS = ["representation", "dataset", "probe", "seed", *SCORE_FIELDS]


def feature_matrices(representation, cfg, ckpt_dir, reps_dir, dataset, seed,
                     X_train, X_test_clean, X_test_corrupt, scaler, cuda):
    """Return (train, clean test, corrupted test) matrices for one representation.

    The input scaler is applied after corruption and was fitted on clean TRAIN, exactly as
    in Parts B and C - a scaler refitted on corrupted data would leak the damage into the
    preprocessing and stop this being a test of the frozen pipeline.
    """
    if representation == "raw":
        return (flatten(apply_scaler(X_train, scaler)),
                flatten(apply_scaler(X_test_clean, scaler)),
                flatten(apply_scaler(X_test_corrupt, scaler)))

    cached = reps_dir / f"{representation}_{dataset}_seed{seed}_TRAIN.npy"
    cached_test = reps_dir / f"{representation}_{dataset}_seed{seed}_TEST.npy"
    if not (cached.exists() and cached_test.exists()):
        return None
    model = load_pretrained(representation, cfg, ckpt_dir, dataset, seed,
                            X_train.shape[1], cuda)
    if model is None:
        return None
    # TRAIN and clean TEST are reused from dev/5_; only the corrupted TEST needs encoding.
    corrupt_test = encode_frozen(representation, model,
                                 apply_scaler(X_test_corrupt, scaler))
    return np.load(cached), np.load(cached_test), corrupt_test


def main() -> int:
    cfg = load_config()
    known = cfg["datasets"]["required"] + cfg["datasets"].get("optional", [])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS,
                        default=list(REPRESENTATIONS))
    parser.add_argument("--datasets", nargs="+", choices=known,
                        default=cfg["datasets"]["required"])
    parser.add_argument("--probes", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", default="corruption.csv")
    args = parser.parse_args()

    cuda = torch.cuda.is_available() and not args.cpu
    ckpt_dir = results_dir(cfg, "checkpoints")
    reps_dir = results_dir(cfg, "representations")
    out_path = results_dir(cfg, "metrics") / args.out
    if out_path.exists():
        out_path.unlink()
    rows: list[dict] = []

    print(f"corruption: {cfg['corruption']['proportion']:.0%} of each series, "
          f"{cfg['corruption']['n_intervals']} contiguous interval, all channels, "
          f"{cfg['corruption']['replacement']}\n")

    for dataset in args.datasets:
        X_train, y_train = load_split(cfg, dataset, "TRAIN")
        X_test, y_test = load_split(cfg, dataset, "TEST")
        scaler = load_scaler(processed_dir(cfg) / f"{dataset}_scaler.json")
        # One corrupted test set per seed, shared by every representation and probe.
        corrupted = {s: corrupt(X_test, scaler, cfg, s) for s in cfg["seeds"]}
        print(f"{dataset}")

        for representation in args.representations:
            for kind in args.probes:
                cleans, corrupts, deltas = [], [], []
                for seed in cfg["seeds"]:
                    mats = feature_matrices(
                        representation, cfg, ckpt_dir, reps_dir, dataset, seed,
                        X_train, X_test, corrupted[seed], scaler, cuda)
                    if mats is None:
                        continue
                    A, B_clean, B_corrupt = mats
                    # Same representation scaler as dev/6_, fitted on clean TRAIN.
                    result = clean_vs_corrupt(kind, A, y_train, B_clean, B_corrupt, y_test,
                                              cfg, seed, standardise=representation != "raw")
                    cleans.append(result["accuracy_clean"])
                    corrupts.append(result["accuracy_corrupt"])
                    deltas.append(result["delta_accuracy"])
                    rows.append({"representation": representation, "dataset": dataset,
                                 "probe": kind, "seed": seed, **result})
                if cleans:
                    c_mu, c_sd = mean_sd(cleans)
                    d_mu, d_sd = mean_sd(corrupts)
                    delta_mu, delta_sd = mean_sd(deltas)
                    print(f"  {representation:<7} {kind:<20} clean {c_mu:.4f}  "
                          f"corrupt {d_mu:.4f}  delta {delta_mu:+.4f}+/-{delta_sd:.4f}")
        print()

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

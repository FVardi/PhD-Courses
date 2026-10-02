"""Part E, exploratory: the masking ablation under heavier and fragmented dropout.

The main Part E result (dev/8_) is the pre-declared 20% x 1-block setting, where continuous
and binomial masking did not differ. That null has two readings this sweep separates:

    too little damage   binomial already loses only ~3 points at 20%; if the arms part
                        at 40%, masking shape matters and 20% was simply too mild.
    wrong gap shape     the continuous training mask is 5 runs of 10% of a crop, not one
                        block. Five shorter test blocks are the best-matched test for it.

The grid (ablation.sweep in src/config.yaml) was declared after the main result was seen,
so this is exploratory and every cell is reported. It never replaces the main result.

Nothing is trained. Each encoder is the one dev/8_ evaluated, and each probe is tuned and
fitted ONCE on clean TRAIN and then scores every corrupted TEST (clean_vs_many), so
differences between settings come from the test arrays alone. The 20% x 1 cell uses the
Part D arrays bit for bit and must reproduce mask_ablation.csv; the script checks it.

Writes results/metrics/mask_sweep.csv.

    python -u dev/8b_ts2vec_mask_sweep.py
"""

import argparse
import csv
import sys
import warnings
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.corruption.dropout import from_config as corrupt  # noqa: E402
from src.dataio.scaling import apply_scaler, load_scaler  # noqa: E402
from src.dataio.splits import load_split, processed_dir  # noqa: E402
from src.evaluation.metrics import mean_sd  # noqa: E402
from src.evaluation.robustness import FIELDS as SCORE_FIELDS, clean_vs_many  # noqa: E402
from src.methods import mask_checkpoint_path, ts2vec  # noqa: E402
from src.utils.config import load_config, results_dir  # noqa: E402

warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)

FIELDS = ["mask", "dataset", "probe", "seed", "proportion", "n_intervals", *SCORE_FIELDS]


def check_against_main(rows: list[dict], main: Path, cfg: dict) -> None:
    """The configured setting must reproduce dev/8_'s rows exactly (same arrays, same fit)."""
    if not main.exists():
        print(f"\n{main.name} not found; skipping the consistency check")
        return
    with main.open(encoding="utf-8") as fh:
        reference = {(r["mask"], r["probe"], int(r["seed"])): r for r in csv.DictReader(fh)}
    spec = cfg["corruption"]
    print(f"\n{spec['proportion']:.0%} x {spec['n_intervals']} vs {main.name}, "
          "|difference| in accuracy:")
    for row in rows:
        if (row["proportion"], row["n_intervals"]) != (spec["proportion"], spec["n_intervals"]):
            continue
        ref = reference.get((row["mask"], row["probe"], row["seed"]))
        if ref is None:
            print(f"  {row['mask']:<10} seed {row['seed']}: no reference row")
            continue
        print(f"  {row['mask']:<10} seed {row['seed']}: "
              f"clean {abs(row['accuracy_clean'] - float(ref['accuracy_clean'])):.4f}  "
              f"corrupt {abs(row['accuracy_corrupt'] - float(ref['accuracy_corrupt'])):.4f}")


def main() -> int:
    cfg = load_config()
    dataset = cfg["ablation"]["dataset"]
    sweep = cfg["ablation"]["sweep"]
    settings = list(product(sweep["proportions"], sweep["n_intervals"]))
    kind = sweep["probe"]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

    ckpt_dir = results_dir(cfg, "checkpoints")
    out_path = results_dir(cfg, "metrics") / "mask_sweep.csv"

    X_train, y_train = load_split(cfg, dataset, "TRAIN")
    X_test, y_test = load_split(cfg, dataset, "TEST")
    scaler = load_scaler(processed_dir(cfg) / f"{dataset}_scaler.json")
    X_train_s = apply_scaler(X_train, scaler)
    X_test_s = apply_scaler(X_test, scaler)

    print(f"{dataset}, probe {kind}, settings (proportion x blocks): "
          + ", ".join(f"{p:.0%}x{n}" for p, n in settings))
    rows: list[dict] = []
    for mask in cfg["ablation"]["masks"]:
        for seed in cfg["seeds"]:
            path = mask_checkpoint_path(ckpt_dir, dataset, seed, mask)
            if not path.exists():
                raise FileNotFoundError(f"{path.name} missing: run dev/8_ first")
            model = ts2vec.build(cfg, input_dims=X_train.shape[1], device=device,
                                 mask_mode=mask)
            ts2vec.load_encoder(model, path)
            A = ts2vec.encode(model, X_train_s)
            B_clean = ts2vec.encode(model, X_test_s)
            # Same seed for every setting, so settings differ only in proportion/blocks.
            B_corrupts = {
                (p, n): ts2vec.encode(model, apply_scaler(
                    corrupt(X_test, scaler, cfg, seed, proportion=p, n_intervals=n), scaler))
                for p, n in settings
            }
            scored = clean_vs_many(kind, A, y_train, B_clean, B_corrupts, y_test,
                                   cfg, seed, standardise=True)
            for (p, n), result in scored.items():
                rows.append({"mask": mask, "dataset": dataset, "probe": kind, "seed": seed,
                             "proportion": p, "n_intervals": n, **result})
            print(f"  {mask:<10} seed {seed}: done")

    print(f"\n{'setting':<9} {'mask':<11} {'corrupt':>16} {'delta':>17}")
    for p, n in settings:
        for mask in cfg["ablation"]["masks"]:
            cell = [r for r in rows if r["mask"] == mask
                    and (r["proportion"], r["n_intervals"]) == (p, n)]
            d_mu, d_sd = mean_sd([r["accuracy_corrupt"] for r in cell])
            delta_mu, delta_sd = mean_sd([r["delta_accuracy"] for r in cell])
            print(f"{p:.0%} x {n:<3} {mask:<11} {d_mu:.4f}+/-{d_sd:.4f}  "
                  f"{delta_mu:+.4f}+/-{delta_sd:.4f}")

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} rows -> {out_path.relative_to(ROOT)}")

    check_against_main(rows, results_dir(cfg, "metrics") / "mask_ablation.csv", cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

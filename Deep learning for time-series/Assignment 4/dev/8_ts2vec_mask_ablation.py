"""Part E: FordA binomial vs continuous timestamp masking, clean and corrupted, matched budget and seeds.

The two arms differ only in TSEncoder.mask_mode. The mask is applied during training only,
after input_fc, and zeroes whole hidden vectors at the chosen timestamps:

    binomial    each timestamp dropped independently with p=0.5 (the TS2Vec default).
                Holes are scattered, so almost every masked point has a visible neighbour.
    continuous  generate_continuous_mask(n=5, l=0.1): 5 runs, each 10% of the crop. Runs
                may overlap, so ~41% is covered in expectation. Gaps are too long to fill
                from neighbours, so the encoder must use more distant context.

Hypothesis: the Part D corruption is one contiguous 20% block, which continuous masking
resembles far more closely, so continuous-masked encoders may lose less under it.

Confounds, stated rather than tuned away (both arms keep the published defaults):
  - amount masked differs (~41% vs 50%), so shape is not the only thing that changes;
  - the training mask zeroes the HIDDEN vector, while the test corruption fills the INPUT
    with the channel mean (0 after scaling), which input_fc maps to its bias, not to 0.
    The two perturbations have similar shapes but are not identical.

Matched design. The binomial arm reuses the Part C checkpoints: same code, same n_iters
(600, TS2Vec's own rule for FordA), same seeds, and seeding before build() means both arms
start from identical initial weights. Only the continuous arm is trained here.

Each continuous run counts calls to the checkout's two mask generators during fit() and
refuses to save unless continuous calls > 0 and binomial calls == 0. That shows which mask
was used from the calls themselves, not from the attribute we set. A smoke run
(--n-iters) also trains a binomial arm, which works as the negative control for the counter.

Evaluation matches Part D exactly (src/evaluation/robustness.py): 100% labels,
StandardScaler on the representation, probes tuned on TRAIN only, scored on clean TEST and
on the same seeded corrupted TEST. The binomial rows must reproduce the TS2Vec FordA rows
of corruption.csv; the script prints that comparison at the end.

Writes:
  results/checkpoints/ts2vec_FordA_seed{seed}_continuous.pth
  results/runs/ts2vec_mask_pretrain.csv
  results/metrics/mask_ablation.csv

    python -u dev/8_ts2vec_mask_ablation.py
    python -u dev/8_ts2vec_mask_ablation.py --n-iters 20 --seeds 42   # quick check
"""

import argparse
import csv
import platform
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.corruption.dropout import from_config as corrupt  # noqa: E402
from src.dataio.scaling import apply_scaler, load_scaler  # noqa: E402
from src.dataio.splits import load_split, processed_dir  # noqa: E402
from src.evaluation.metrics import mean_sd  # noqa: E402
from src.evaluation.robustness import FIELDS as SCORE_FIELDS, clean_vs_corrupt  # noqa: E402
from src.methods import checkpoint_path as part_c_checkpoint, ts2vec  # noqa: E402
from src.probes.probes import KINDS  # noqa: E402
from src.utils.config import load_config, results_dir  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)

FIELDS = ["mask", "dataset", "probe", "seed", *SCORE_FIELDS]
RUN_FIELDS = [
    "mask", "dataset", "seed", "n_iters", "calls_binomial", "calls_continuous",
    "device", "seconds", "checkpoint",
]


def checkpoint(ckpt_dir: Path, dataset: str, seed: int, mask: str,
               n_iters: int | None) -> Path:
    """Protocol binomial -> the Part C file. Everything else gets its own name, and smoke
    runs are tagged so a short encoder can never be reused as a protocol run."""
    if mask == "binomial" and n_iters is None:
        return part_c_checkpoint(ckpt_dir, "ts2vec", dataset, seed)
    tag = "" if n_iters is None else f"_smoke{n_iters}"
    return ckpt_dir / f"ts2vec_{dataset}_seed{seed}_{mask}{tag}.pth"


def append_run(path: Path, row: dict) -> None:
    """Append one run immediately, so an interrupted job keeps the record of finished runs."""
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def train_arm(cfg, X, dataset, seed, mask, iters, path, device, label, runs_path):
    """Pretrain one encoder with `mask`, verify the mask from the generator calls, save."""
    # Seed BEFORE build(): the encoder is initialised in the constructor, so both arms
    # of a seed start from the same weights.
    set_seed(seed)
    model = ts2vec.build(cfg, input_dims=X.shape[1], device=device, mask_mode=mask)
    t0 = time.time()
    with ts2vec.count_mask_calls(model) as calls:
        ts2vec.pretrain(model, X, n_iters=iters)
    elapsed = time.time() - t0

    other = "binomial" if mask == "continuous" else "continuous"
    if calls[mask] == 0 or calls[other] != 0:
        raise RuntimeError(f"mask switch did not take effect for {mask}: calls={calls}")
    ts2vec.save_encoder(model, path)

    print(f"  {mask:<10} seed {seed}: {iters} iters in {elapsed:.0f} s, "
          f"mask calls {calls} -> {path.name}")
    append_run(runs_path, {
        "mask": mask, "dataset": dataset, "seed": seed, "n_iters": iters,
        "calls_binomial": calls["binomial"], "calls_continuous": calls["continuous"],
        "device": label, "seconds": round(elapsed, 1),
        "checkpoint": str(path.relative_to(ROOT)),
    })


def compare_with_part_d(rows: list[dict], part_d: Path, dataset: str) -> None:
    """Print how far the binomial arm is from the Part D TS2Vec rows, per probe and seed."""
    if not part_d.exists():
        print(f"\n{part_d.name} not found; skipping the Part D consistency check")
        return
    with part_d.open(encoding="utf-8") as fh:
        reference = {(r["probe"], int(r["seed"])): r for r in csv.DictReader(fh)
                     if r["representation"] == "ts2vec" and r["dataset"] == dataset}
    print(f"\nbinomial arm vs {part_d.name} (ts2vec, {dataset}), |difference| in accuracy:")
    for row in rows:
        if row["mask"] != "binomial":
            continue
        ref = reference.get((row["probe"], row["seed"]))
        if ref is None:
            print(f"  {row['probe']:<20} seed {row['seed']}: no Part D row")
            continue
        d_clean = abs(row["accuracy_clean"] - float(ref["accuracy_clean"]))
        d_dirty = abs(row["accuracy_corrupt"] - float(ref["accuracy_corrupt"]))
        print(f"  {row['probe']:<20} seed {row['seed']}: "
              f"clean {d_clean:.4f}  corrupt {d_dirty:.4f}")


def main() -> int:
    cfg = load_config()
    dataset = cfg["ablation"]["dataset"]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--masks", nargs="+", choices=ts2vec.MASK_MODES,
                        default=cfg["ablation"]["masks"])
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    parser.add_argument("--probes", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--n-iters", type=int, default=None,
                        help="override TS2Vec's own rule (for smoke tests only)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="retrain encoders trained by this script (never Part C's)")
    args = parser.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    label = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu"
    print(f"device: {label}  ({device})")
    smoke = "" if args.n_iters is None else f"_smoke{args.n_iters}"
    if smoke:
        print(f"WARNING: n_iters overridden to {args.n_iters}; not a protocol run")

    ckpt_dir = results_dir(cfg, "checkpoints")
    runs_path = results_dir(cfg, "runs") / f"ts2vec_mask_pretrain{smoke}.csv"
    out_path = results_dir(cfg, "metrics") / f"mask_ablation{smoke}.csv"

    X_train, y_train = load_split(cfg, dataset, "TRAIN")
    X_test, y_test = load_split(cfg, dataset, "TEST")
    scaler = load_scaler(processed_dir(cfg) / f"{dataset}_scaler.json")
    X_train_s = apply_scaler(X_train, scaler)
    X_test_s = apply_scaler(X_test, scaler)
    iters = args.n_iters if args.n_iters is not None else ts2vec.default_iters(X_train_s)

    # --- pretraining -------------------------------------------------------------------
    print(f"\npretraining {dataset}  {X_train_s.shape}  n_iters={iters}")
    for mask in args.masks:
        for seed in args.seeds:
            path = checkpoint(ckpt_dir, dataset, seed, mask, args.n_iters)
            reused = mask == "binomial" and args.n_iters is None
            if reused:
                if not path.exists():
                    raise FileNotFoundError(
                        f"{path.name} missing: the binomial arm reuses Part C, "
                        "run dev/4_pretrain_ts2vec.py first")
                print(f"  {mask:<10} seed {seed}: reusing Part C {path.name}")
            elif path.exists() and not args.force:
                print(f"  {mask:<10} seed {seed}: checkpoint exists, skipping")
            else:
                train_arm(cfg, X_train_s, dataset, seed, mask, iters, path,
                          device, label, runs_path)

    # --- evaluation --------------------------------------------------------------------
    # Corruption precedes the input scaler, as in Part D, and is seeded by the same seed,
    # so these are the exact arrays Part D scored.
    print(f"\nevaluating (corruption: {cfg['corruption']['proportion']:.0%}, "
          f"{cfg['corruption']['n_intervals']} contiguous interval)")
    rows: list[dict] = []
    for mask in args.masks:
        for seed in args.seeds:
            X_corrupt_s = apply_scaler(corrupt(X_test, scaler, cfg, seed), scaler)
            model = ts2vec.build(cfg, input_dims=X_train.shape[1], device=device,
                                 mask_mode=mask)
            ts2vec.load_encoder(model, checkpoint(ckpt_dir, dataset, seed, mask,
                                                  args.n_iters))
            A = ts2vec.encode(model, X_train_s)
            B_clean = ts2vec.encode(model, X_test_s)
            B_corrupt = ts2vec.encode(model, X_corrupt_s)
            for kind in args.probes:
                result = clean_vs_corrupt(kind, A, y_train, B_clean, B_corrupt, y_test,
                                          cfg, seed, standardise=True)
                rows.append({"mask": mask, "dataset": dataset, "probe": kind,
                             "seed": seed, **result})

    print()
    for kind in args.probes:
        for mask in args.masks:
            cell = [r for r in rows if r["mask"] == mask and r["probe"] == kind]
            c_mu, c_sd = mean_sd([r["accuracy_clean"] for r in cell])
            d_mu, d_sd = mean_sd([r["accuracy_corrupt"] for r in cell])
            delta_mu, delta_sd = mean_sd([r["delta_accuracy"] for r in cell])
            print(f"  {mask:<10} {kind:<20} clean {c_mu:.4f}+/-{c_sd:.4f}  "
                  f"corrupt {d_mu:.4f}+/-{d_sd:.4f}  delta {delta_mu:+.4f}+/-{delta_sd:.4f}")

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} rows -> {out_path.relative_to(ROOT)}")

    if not smoke:
        compare_with_part_d(rows, results_dir(cfg, "metrics") / "corruption.csv", dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())

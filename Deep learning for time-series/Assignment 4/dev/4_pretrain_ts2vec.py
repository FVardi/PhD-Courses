"""Part C: TS2Vec pretraining on the unlabelled official training series.

One encoder per (dataset, seed) - nine in total - with the checkout's published defaults
(output_dims=320, hidden_dims=64, depth=10, lr=0.001, batch_size=16). Only input_dims,
which must match the data, and the device are set by us.

n_iters follows TS2Vec's own rule (200 for a training array of at most 100000 elements,
600 otherwise), computed explicitly so it lands in the run record rather than staying
implicit. For these datasets that is 600 for FordA and AWR, 200 for Epilepsy - Epilepsy
trains a third as long as the others by the method's own default, which is worth stating
rather than describing the budget as a single number.

Labels are never involved. Pretraining only: extraction is dev/5_extract_representations.py
and probe fitting is dev/6_probe_eval.py, kept separate so the frozen encoder is visible in
the pipeline's shape.

Writes:
  results/checkpoints/ts2vec_{dataset}_seed{seed}.pth
  results/runs/ts2vec_pretrain.csv

    python dev/4_pretrain_ts2vec.py
    python dev/4_pretrain_ts2vec.py --datasets Epilepsy --n-iters 50   # quick check
"""

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.dataio.scaling import apply_scaler, load_scaler  # noqa: E402
from src.dataio.splits import load_split, processed_dir  # noqa: E402
from src.methods import ts2vec  # noqa: E402
from src.utils.config import load_config, results_dir  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

FIELDS = [
    "method", "dataset", "seed", "n_series", "channels", "length",
    "n_iters", "batch_size", "output_dims", "device", "seconds", "checkpoint",
]


def append_run(path: Path, row: dict) -> None:
    """Append one run immediately.

    Written per run rather than at the end: a long job that is interrupted must not lose
    the record of the work it already finished.
    """
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def checkpoint_path(directory: Path, dataset: str, seed: int, override: int | None) -> Path:
    """Smoke runs get their own filename so a short encoder can never be reused as a
    protocol run."""
    tag = "" if override is None else f"_smoke{override}"
    return directory / f"ts2vec_{dataset}_seed{seed}{tag}.pth"


def main() -> int:
    cfg = load_config()
    known = cfg["datasets"]["required"] + cfg["datasets"].get("optional", [])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=known,
                        default=cfg["datasets"]["required"])
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    parser.add_argument("--n-iters", type=int, default=None,
                        help="override TS2Vec's own rule (for smoke tests only)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    label = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu"
    print(f"device: {label}  ({device})")
    if args.n_iters is not None:
        print(f"WARNING: n_iters overridden to {args.n_iters}; not a protocol run")

    ckpt_dir = results_dir(cfg, "checkpoints")
    runs_path = results_dir(cfg, "runs") / "ts2vec_pretrain.csv"
    rows: list[dict] = []

    for name in args.datasets:
        X, _ = load_split(cfg, name, "TRAIN")
        X = apply_scaler(X, load_scaler(processed_dir(cfg) / f"{name}_scaler.json"))
        n, channels, length = X.shape
        iters = args.n_iters if args.n_iters is not None else ts2vec.default_iters(X)
        print(f"\n{name}  {X.shape}  n_iters={iters}")

        for seed in args.seeds:
            path = checkpoint_path(ckpt_dir, name, seed, args.n_iters)
            if path.exists() and not args.force:
                print(f"  seed {seed}: checkpoint exists, skipping")
                continue

            # Seed BEFORE build(): TS2Vec creates its encoder in the constructor.
            set_seed(seed)
            model = ts2vec.build(cfg, input_dims=channels, device=device)

            t0 = time.time()
            ts2vec.pretrain(model, X, n_iters=iters)
            elapsed = time.time() - t0
            ts2vec.save_encoder(model, path)

            print(f"  seed {seed}: {iters} iters in {elapsed / 60:.1f} min -> {path.name}")
            row = {
                "method": "ts2vec", "dataset": name, "seed": seed, "n_series": n,
                "channels": channels, "length": length, "n_iters": iters,
                "batch_size": model.batch_size, "output_dims": model._net.output_dims,
                "device": label, "seconds": round(elapsed, 1),
                "checkpoint": str(path.relative_to(ROOT)),
            }
            append_run(runs_path, row)
            rows.append(row)

    if rows:
        print(f"\n{len(rows)} run(s) recorded -> {runs_path.relative_to(ROOT)}")
    else:
        print("\nnothing to do (use --force to retrain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

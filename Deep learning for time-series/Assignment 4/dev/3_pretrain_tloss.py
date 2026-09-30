"""Part C: T-Loss pretraining on the unlabelled official training series.

One encoder per (dataset, seed) - nine in total - trained with the checkout's published
hyperparameters, unchanged except for in_channels (which must match the data) and the
cuda/gpu flags. Labels are never touched: fit_encoder() is called with y=None, which also
disables the early-stopping heuristic, matching the published early_stopping: null.

Pretraining only. Representation extraction is dev/5_extract_representations.py and probe
fitting is dev/6_probe_eval.py, kept separate so that "the encoder is frozen" is visible in
the shape of the pipeline rather than asserted.

Writes:
  results/checkpoints/tloss_{dataset}_seed{seed}_CausalCNN_encoder.pth
  results/runs/tloss_pretrain.csv    runtime, device and the resolved hyperparameters

Re-running skips datasets whose checkpoint already exists; --force retrains.

    python dev/3_pretrain_tloss.py
    python dev/3_pretrain_tloss.py --datasets Epilepsy --nb-steps 50   # quick check
"""

import argparse
import csv
import json
import platform
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.dataio.scaling import apply_scaler, load_scaler  # noqa: E402
from src.dataio.splits import load_split, processed_dir  # noqa: E402
from src.methods import tloss  # noqa: E402
from src.utils.config import load_config, results_dir  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

# The checkout targets an older torch; its weight_norm call is deprecated but not broken.
warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)

FIELDS = [
    "method", "dataset", "seed", "n_series", "channels", "length",
    "nb_steps", "batch_size", "out_channels", "device", "seconds", "checkpoint",
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

def checkpoint_prefix(directory: Path, dataset: str, seed: int, nb_steps: int | None) -> Path:
    """Smoke runs get their own filename so a short encoder can never be mistaken for -
    or silently reused as - a protocol run."""
    tag = "" if nb_steps is None else f"_smoke{nb_steps}"
    return directory / f"tloss_{dataset}_seed{seed}{tag}"


def main() -> int:
    cfg = load_config()
    known = cfg["datasets"]["required"] + cfg["datasets"].get("optional", [])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=known,
                        default=cfg["datasets"]["required"])
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    parser.add_argument("--nb-steps", type=int, default=None,
                        help="override the published nb_steps (for smoke tests only)")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    parser.add_argument("--force", action="store_true", help="retrain over existing checkpoints")
    args = parser.parse_args()

    cuda = torch.cuda.is_available() and not args.cpu
    device = torch.cuda.get_device_name(0) if cuda else platform.processor() or "cpu"
    print(f"device: {device}  (cuda={cuda})")
    if args.nb_steps is not None:
        print(f"WARNING: nb_steps overridden to {args.nb_steps}; not a protocol run")

    ckpt_dir = results_dir(cfg, "checkpoints")
    runs_path = results_dir(cfg, "runs") / "tloss_pretrain.csv"
    rows: list[dict] = []

    for name in args.datasets:
        X, _ = load_split(cfg, name, "TRAIN")
        X = apply_scaler(X, load_scaler(processed_dir(cfg) / f"{name}_scaler.json"))
        n, channels, length = X.shape
        print(f"\n{name}  {X.shape}")

        for seed in args.seeds:
            prefix = checkpoint_prefix(ckpt_dir, name, seed, args.nb_steps)
            final = Path(f"{prefix}_{tloss.ARCHITECTURE}_encoder.pth")
            if final.exists() and not args.force:
                print(f"  seed {seed}: checkpoint exists, skipping")
                continue

            # Seed BEFORE build(): the encoder's weights are initialised in the
            # constructor, and fit_encoder draws its batches afterwards.
            set_seed(seed)
            model = tloss.build(cfg, in_channels=channels, cuda=cuda)
            params = model.get_params()
            if args.nb_steps is not None:
                model.set_params(**{**params, "nb_steps": args.nb_steps})
                params = model.get_params()

            t0 = time.time()
            tloss.pretrain(model, X)
            elapsed = time.time() - t0
            tloss.save_encoder(model, prefix)

            print(f"  seed {seed}: {params['nb_steps']} steps in {elapsed / 60:.1f} min "
                  f"-> {final.name}")
            row = {
                "method": "tloss", "dataset": name, "seed": seed, "n_series": n,
                "channels": channels, "length": length, "nb_steps": params["nb_steps"],
                "batch_size": params["batch_size"], "out_channels": params["out_channels"],
                "device": device, "seconds": round(elapsed, 1),
                "checkpoint": str(final.relative_to(ROOT)),
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

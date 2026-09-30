"""Part C: extract frozen representations from the pretrained encoders.

Loads each checkpoint written by dev/3_ and dev/4_, applies it to the full official TRAIN
and TEST splits, and caches one (n_series, 320) array per split. Nothing is trained here and
no label is read: this script only ever runs an encoder forward.

Keeping extraction separate from probe fitting is what makes "the encoder is frozen"
checkable rather than asserted - dev/6_probe_eval.py loads .npy files and never sees an
encoder at all, so it cannot update one.

Clean data only. Part D's corrupted test sets are encoded by dev/7_corruption_eval.py, which
owns the corruption convention; keeping it out of here leaves this script a pure forward pass.

Both splits are encoded in full. The 10% label regime selects rows of the cached TRAIN array
afterwards, so the 10% and 100% probes are guaranteed to see representations from an
identical encoder pass.

Writes:
  results/representations/{method}_{dataset}_seed{seed}_{SPLIT}.npy
  results/runs/extract.csv

Safe to re-run as more checkpoints appear; existing outputs are skipped unless --force.

    python dev/5_extract_representations.py
    python dev/5_extract_representations.py --methods ts2vec
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.dataio.scaling import apply_scaler, load_scaler  # noqa: E402
from src.dataio.splits import SPLITS, load_split, processed_dir  # noqa: E402
from src.methods import tloss, ts2vec  # noqa: E402
from src.utils.config import load_config, results_dir  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)

METHODS = ("tloss", "ts2vec")
FIELDS = ["method", "dataset", "seed", "split", "n_series", "dims", "seconds", "path"]


def tloss_checkpoint(ckpt_dir: Path, dataset: str, seed: int) -> tuple[Path, Path]:
    """T-Loss saves as '{prefix}_{architecture}_encoder.pth'; it loads by prefix."""
    prefix = ckpt_dir / f"tloss_{dataset}_seed{seed}"
    return prefix, Path(f"{prefix}_{tloss.ARCHITECTURE}_encoder.pth")


def load_frozen(method: str, cfg: dict, ckpt_dir: Path, dataset: str, seed: int,
                channels: int, cuda: bool):
    """Rebuild the architecture and restore the trained weights. Returns None if absent."""
    if method == "tloss":
        prefix, final = tloss_checkpoint(ckpt_dir, dataset, seed)
        if not final.exists():
            return None
        model = tloss.build(cfg, in_channels=channels, cuda=cuda)
        return tloss.load_encoder(model, prefix)

    path = ckpt_dir / f"ts2vec_{dataset}_seed{seed}.pth"
    if not path.exists():
        return None
    model = ts2vec.build(cfg, input_dims=channels, device="cuda" if cuda else "cpu")
    return ts2vec.load_encoder(model, path)


def main() -> int:
    cfg = load_config()
    known = cfg["datasets"]["required"] + cfg["datasets"].get("optional", [])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=known,
                        default=cfg["datasets"]["required"])
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cuda = torch.cuda.is_available() and not args.cpu
    ckpt_dir = results_dir(cfg, "checkpoints")
    out_dir = results_dir(cfg, "representations")
    runs_path = results_dir(cfg, "runs") / "extract.csv"
    written = missing = 0

    for method in args.methods:
        for dataset in args.datasets:
            scaler = load_scaler(processed_dir(cfg) / f"{dataset}_scaler.json")
            splits = {}
            for split in SPLITS:
                X, _ = load_split(cfg, dataset, split)
                splits[split] = apply_scaler(X, scaler)
            channels = splits["TRAIN"].shape[1]

            for seed in args.seeds:
                targets = {s: out_dir / f"{method}_{dataset}_seed{seed}_{s}.npy"
                           for s in SPLITS}
                if all(p.exists() for p in targets.values()) and not args.force:
                    continue

                set_seed(seed)          # the checkpoint overwrites these weights immediately
                model = load_frozen(method, cfg, ckpt_dir, dataset, seed, channels, cuda)
                if model is None:
                    print(f"  {method:<7} {dataset:<28} seed {seed}: no checkpoint yet")
                    missing += 1
                    continue

                for split, X in splits.items():
                    t0 = time.time()
                    Z = (tloss.encode(model, X) if method == "tloss"
                         else ts2vec.encode(model, X))
                    elapsed = time.time() - t0
                    np.save(targets[split], Z)
                    written += 1
                    print(f"  {method:<7} {dataset:<28} seed {seed} {split:<5} "
                          f"-> {Z.shape}  {elapsed:.1f}s")
                    is_new = not runs_path.exists()
                    with runs_path.open("a", newline="", encoding="utf-8") as fh:
                        writer = csv.DictWriter(fh, fieldnames=FIELDS)
                        if is_new:
                            writer.writeheader()
                        writer.writerow({
                            "method": method, "dataset": dataset, "seed": seed,
                            "split": split, "n_series": Z.shape[0], "dims": Z.shape[1],
                            "seconds": round(elapsed, 2),
                            "path": str(targets[split].relative_to(ROOT)),
                        })

    print(f"\n{written} array(s) written; {missing} (method, dataset, seed) still unpretrained")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Part F: drive the unchanged TF-C checkout for FD-A pretraining then FD-B fine-tune/test; record the run.

This part does not use the project's own protocol. It runs TF-C's supplied execution path
as shipped - its data loading, its subset, its loss, its epoch counts, its evaluation - and
documents what that path actually does. Two stages, each one call of the checkout's main.py:

    pre_train        FD-A train (labels unused)  ──► ckp_last.pt
    fine_tune_test   ckp_last.pt + FD-B train (60 labelled) ──► FD-B test, every epoch

What is NOT changed, although the project would do it differently elsewhere:
  - main.py hard-codes subset=True, so pre-training sees 10 batches' worth of FD-A, not all;
  - the reported result is the epoch with the best TEST accuracy (no validation set is
    used). It is optimistic, and it is the figure TF-C's own procedure reports, so it is the
    figure recorded here as the result.

What is added from outside, without editing the checkout (see src/methods/tfc.py):
  - directory junctions datasets/FD_A, datasets/FD_B -> data/raw/FD-A, data/raw/FD-B;
  - one in-memory shim, np.float = float, for NumPy >= 1.24;
  - --logs_save_dir, so logs and the pre-training checkpoint land in results/tfc/. The
    fine-tuning stage still writes its best model to experiments_logs/finetunemodel/ under
    code/TFC: that path is hard-coded in trainer.py. The checkout ignores *.pt, so its
    working tree stays clean.

Writes results/tfc/execution_record.json plus the two stage logs.

    python -u dev/9_tfc_fda_fdb.py
    python -u dev/9_tfc_fda_fdb.py --stages fine_tune_test    # reuse the saved checkpoint
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402

from src.methods import tfc  # noqa: E402
from src.utils.config import load_config, results_dir  # noqa: E402

STAGES = ("pre_train", "fine_tune_test")


def git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(checkout), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def environment() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": numpy.__version__, "sklearn": sklearn.__version__,
        "os": platform.platform(), "cpu": platform.processor(),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if cuda else None,
        "cuda": torch.version.cuda if cuda else None,
    }


def split_sizes(cfg: dict) -> dict:
    """Series available in each file main.py opens, to set beside what it actually keeps."""
    sizes = {}
    for key in ("fd_a", "fd_b"):
        for split in ("train", "test"):
            blob = torch.load(Path(cfg["tfc_data"][key]) / f"{split}.pt")
            sizes[f"{key}_{split}"] = int(blob["labels"].shape[0])
    return sizes


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args()

    checkout = Path(cfg["checkouts"]["tfc"]["path"])
    out_dir = results_dir(cfg, "tfc")
    record_path = out_dir / "execution_record.json"
    # Keep an earlier stage's entry when only one stage is rerun.
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}

    commit = git(checkout, "rev-parse", "--short", "HEAD")
    if not commit.startswith(cfg["checkouts"]["tfc"]["commit"]):
        raise RuntimeError(f"TF-C checkout is at {commit}, expected "
                           f"{cfg['checkouts']['tfc']['commit']}")
    record.update({
        "checkout": {"path": str(checkout), "commit": commit,
                     "tracked_files_modified": git(checkout, "status", "--short",
                                                   "--untracked-files=no") != ""},
        "seed": cfg["tfc_run"]["seed"],
        "environment": environment(),
        "shims": tfc.SHIMS,
        "data_links": tfc.ensure_data_links(cfg),
        "data_available": split_sizes(cfg),
    })
    record.setdefault("stages", {})

    for mode in args.stages:
        print(f"\n===== {mode} =====", flush=True)
        log_path = out_dir / f"{mode}.log"
        stage = tfc.run_stage(cfg, mode, out_dir, log_path)
        stage["finished"] = datetime.now().isoformat(timespec="seconds")
        text = log_path.read_text(encoding="utf-8")
        stage["subset_sizes_reported"] = tfc.parse_subset_sizes(text)
        if mode == "fine_tune_test":
            stage["results_percent"] = tfc.parse_results(text)
        record["stages"][mode] = stage
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if stage["returncode"] != 0:
            print(f"\n{mode} failed with return code {stage['returncode']}; see {log_path}")
            return stage["returncode"]
        print(f"\n{mode}: {stage['seconds']:.0f} s")

    checkpoints = sorted(out_dir.rglob("ckp_last.pt"))
    record["checkpoint"] = str(checkpoints[-1]) if checkpoints else None
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    best = record["stages"].get("fine_tune_test", {}).get("results_percent", {}).get("best")
    if best:
        print(f"\nTF-C reported result (best test epoch): accuracy {best['accuracy']:.2f}%, "
              f"macro-F1 {best['macro_f1']:.2f}%")
    print(f"record -> {record_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

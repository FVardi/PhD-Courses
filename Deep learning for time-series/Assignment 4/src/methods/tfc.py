"""Driver-side helpers for the TF-C checkout (Part F).

Unlike T-Loss and TS2Vec, TF-C is not wrapped as a library: Part F runs the checkout's own
main.py end to end, with its own data loading, training loop and evaluation, and records
what happened. Nothing here touches the method. The helpers only

  - link the data into the place main.py hard-codes (ensure_data_links),
  - start main.py from its own folder through the compatibility launcher (run_stage),
  - read the numbers main.py prints (parse_*), since it returns nothing and writes its
    metrics only to stdout.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

from src.utils.config import ROOT

LAUNCHER = Path(__file__).with_name("_tfc_launch.py")
SHIMS = ["np.float = float (alias removed in NumPy 1.24; used by trainer.py's AUROC fallback)"]

# AUROC is printed as "nan" whenever a test batch lacks a class, so numbers may be nan.
_NUM = r"([\d.]+|nan)"
_METRICS = (rf"Acc={_NUM}\|\s*Precision = {_NUM} \| Recall = {_NUM} \| F1 = {_NUM} "
            rf"\| AUROC= {_NUM}\s*\| AUPRC={_NUM}")
_KEYS = ("accuracy", "precision", "recall", "macro_f1", "auroc", "auprc")


def code_dir(cfg: dict) -> Path:
    return Path(cfg["checkouts"]["tfc"]["path"]) / "code" / "TFC"


def ensure_data_links(cfg: dict) -> dict[str, str]:
    """Make datasets/FD_A and datasets/FD_B in the checkout point at data/raw.

    main.py opens ../../datasets/<name> with no way to override it, and <name> must use
    underscores because the same argument selects config_files/<name>_Configs.py. A
    directory junction satisfies that without copying 1.1 GB into the checkout; the
    checkout's .gitignore already ignores datasets/, so its working tree stays clean.
    Returns {link: target} for the execution record.
    """
    datasets = Path(cfg["checkouts"]["tfc"]["path"]) / "datasets"
    links = {}
    for name, folder in cfg["tfc_data"]["raw"].items():
        target = ROOT / cfg["paths"]["data_raw"] / folder
        if not (target / "train.pt").exists():
            raise FileNotFoundError(f"{target} has no train.pt; download {folder} first")
        link = datasets / name
        if not link.exists():
            if sys.platform == "win32":  # junctions need no admin rights, symlinks do
                subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                               check=True, capture_output=True)
            else:
                link.symlink_to(target, target_is_directory=True)
        links[str(link)] = str(target)
    return links


def stage_args(cfg: dict, mode: str, logs_dir: Path) -> list[str]:
    """main.py's own arguments for one stage, exactly as its README gives them, plus the
    seed and a logs directory outside the checkout."""
    run = cfg["tfc_run"]
    # The config uses the datasets' published names (FD-A); main.py needs FD_A, because it
    # builds an import statement from the argument (config_files.FD_A_Configs).
    return ["--training_mode", mode,
            "--pretrain_dataset", run["pretrain_dataset"].replace("-", "_"),
            "--target_dataset", run["target_dataset"].replace("-", "_"),
            "--seed", str(run["seed"]),
            "--logs_save_dir", str(logs_dir)]


def run_stage(cfg: dict, mode: str, logs_dir: Path, log_path: Path) -> dict:
    """Run one stage to completion, streaming its output to the console and to log_path.

    Returns the command, working directory, wall-clock seconds and return code. stderr is
    merged into stdout so the log holds everything in the order it was printed.
    """
    command = [sys.executable, "-u", str(LAUNCHER), *stage_args(cfg, mode, logs_dir)]
    cwd = code_dir(cfg)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
    return {"command": command, "equivalent": "python main.py " + " ".join(command[3:]),
            "cwd": str(cwd), "seconds": round(time.time() - t0, 1),
            "returncode": proc.returncode, "log": str(log_path)}


def parse_subset_sizes(text: str) -> list[int]:
    """Sizes main.py reports for the subsets it keeps (pre-training set, then fine-tuning)."""
    return [int(n) for n in re.findall(r"Using subset for debugging, the datasize is: (\d+)", text)]


def parse_results(text: str) -> dict:
    """The fine-tuning stage's printed metrics.

    best     'Best Testing Performance': the epoch with the highest TEST accuracy. This is
             the figure TF-C's own procedure reports, and it is selected on the test set.
    epochs   the MLP test metrics of every epoch, in order, for the record.
    """
    def row(match):
        return {k: None if v == "nan" else float(v) for k, v in zip(_KEYS, match)}

    best = re.search(r"Best Testing Performance: " + _METRICS, text)
    knn = re.search(r"Best KNN F1 ([\d.]+)", text)
    return {
        "best": row(best.groups()) if best else None,
        "best_knn_macro_f1": float(knn.group(1)) if knn else None,
        "epochs": [row(m) for m in re.findall(r"MLP Testing: " + _METRICS, text)],
    }

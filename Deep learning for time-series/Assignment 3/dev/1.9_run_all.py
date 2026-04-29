"""Run all configurations across all seeds.

Trains and evaluates every (approach, model, seed) combination, then saves
a results CSV to src/results/all_results_{DATASET}.csv.

Skips runs where the predictions parquet already exists (resumable).

Usage:
    python 1.9_run_all.py [--dataset FD001]
    python 1.9_run_all.py --dataset FD002
"""

import argparse
import importlib.util
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SRC_DIR = Path(__file__).parents[1] / "src"
DEV_DIR = Path(__file__).parent


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, DEV_DIR / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_train_mod = _load_module("train",            "1.5_train.py")
_eval_mod  = _load_module("evaluate",         "1.6_evaluate.py")
_xgb_mod   = _load_module("xgboost_baseline", "1.8_xgboost_baseline.py")

DL_CONFIGS = [
    ("window",   "rnn"),
    ("window",   "lstm"),
    ("window",   "tcn"),
    ("sequence", "rnn"),
    ("sequence", "lstm"),
    ("sequence", "tcn"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="FD001")
args, _ = parser.parse_known_args()
DATASET = args.dataset

with open(SRC_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

prefix    = "" if DATASET == "FD001" else f"{DATASET}_"
seeds     = cfg["train_seeds"]
preds_dir = SRC_DIR / "results" / "predictions"
results   = []

all_runs = (
    [(approach, model, seed) for approach, model in DL_CONFIGS for seed in seeds]
    + [("sequence", "xgboost", seed) for seed in seeds]
)
n_total = len(all_runs)

print(f"Dataset: {DATASET}")
print(f"Seeds: {seeds}")
print(f"Total runs: {n_total}  ({len(DL_CONFIGS)} DL configs + XGBoost  ×  {len(seeds)} seeds)")
print("=" * 60)

bar = tqdm(all_runs, ncols=80, unit="run")

for approach, model, seed in bar:
    tag      = f"{approach}/{model}/seed{seed}"
    bar.set_description(f"{approach}/{model}/s{seed}")

    if model == "xgboost":
        out_path = preds_dir / f"{prefix}sequence_xgboost_seed{seed}.parquet"
    else:
        out_path = preds_dir / f"{prefix}{approach}_{model}_seed{seed}.parquet"

    if out_path.exists():
        tqdm.write(f"[skip] {tag}")
        if model == "xgboost":
            rmse, score = _xgb_mod.run(seed, dataset=DATASET)
        else:
            rmse, score = _eval_mod.evaluate(approach, model, seed, dataset=DATASET)
    else:
        tqdm.write(f"[run]  {tag}")
        if model == "xgboost":
            rmse, score = _xgb_mod.run(seed, dataset=DATASET)
        else:
            _train_mod.train(approach, model, seed, dataset=DATASET)
            rmse, score = _eval_mod.evaluate(approach, model, seed, dataset=DATASET)

    results.append({
        "approach": approach,
        "model":    model,
        "seed":     seed,
        "rmse":     rmse,
        "nasa":     score,
    })
    tqdm.write(f"       RMSE={rmse:.4f}  NASA={score:.2f}")

# --- save ---
results_df = pd.DataFrame(results)
out_csv    = SRC_DIR / "results" / f"all_results_{DATASET}.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nAll results saved to {out_csv}")
print(results_df.to_string(index=False))

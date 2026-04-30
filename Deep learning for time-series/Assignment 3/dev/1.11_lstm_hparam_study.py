"""LSTM hyperparameter study.

Grid:
  window_size  : [20, 30, 50]
  hidden_size  : [64, 128]
  learning_rate: [1e-3, 1e-4]

Each configuration is run with 3 seeds. Best configuration per approach
is selected on mean validation RMSE, then evaluated on the test set.

Usage:
    python 1.11_lstm_hparam_study.py
    python 1.11_lstm_hparam_study.py --dataset FD002
"""

import argparse
import copy
import importlib.util
import itertools
import numpy as np
import yaml
import pandas as pd
from pathlib import Path

SRC_DIR = Path(__file__).parents[1] / "src"
DEV_DIR = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="FD001")
args, _ = parser.parse_known_args()
DATASET = args.dataset

STUDY_SEEDS  = [42, 123, 456]
WINDOW_SIZES = [20, 30, 50]
HIDDEN_SIZES = [64, 128]
LR_VALUES    = [1e-3, 1e-4]
APPROACHES   = ["window", "sequence"]


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, DEV_DIR / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_train_mod = _load_module("train",    "1.5_train.py")
_eval_mod  = _load_module("evaluate", "1.6_evaluate.py")

with open(SRC_DIR / "config.yaml") as f:
    base_cfg = yaml.safe_load(f)


def config_id(w, h, lr):
    lr_tag = f"1e{int(round(-np.log10(lr)))}"
    return f"w{w}_h{h}_lr{lr_tag}"


def ckpt_name(approach, w, h, lr, seed):
    return f"study_lstm_{approach}_{config_id(w, h, lr)}_seed{seed}.pt"


# =============================================================================
# Training sweep
# =============================================================================
prefix    = "" if DATASET == "FD001" else f"{DATASET}_"
ckpt_dir  = SRC_DIR / "results" / "checkpoints"
grid      = list(itertools.product(APPROACHES, WINDOW_SIZES, HIDDEN_SIZES, LR_VALUES))
n_total   = len(grid) * len(STUDY_SEEDS)

print(f"Dataset: {DATASET}")
print(f"Grid: {len(grid)} configs × {len(STUDY_SEEDS)} seeds = {n_total} runs\n")

records = []
for approach, w, h, lr in grid:
    cid = config_id(w, h, lr)
    for seed in STUDY_SEEDS:
        cname = f"{prefix}{ckpt_name(approach, w, h, lr, seed)}"
        ckpt_path = ckpt_dir / cname

        if ckpt_path.exists():
            ckpt = __import__("torch").load(ckpt_path, map_location="cpu")
            val_rmse = float(ckpt["val_rmse"])
            print(f"[skip] {approach}/{cid}/seed{seed}  val RMSE={val_rmse:.4f}")
        else:
            print(f"\n[run]  {approach}/{cid}/seed{seed}")
            cfg = copy.deepcopy(base_cfg)
            cfg["window_size"]          = w
            cfg["training"]["hidden_size"] = h
            cfg["training"]["lr"]          = lr

            val_rmse = _train_mod.train(
                approach, "lstm", seed=seed, cfg=cfg, ckpt_name=cname, dataset=DATASET
            )

        records.append({
            "approach":    approach,
            "config_id":   cid,
            "window_size": w,
            "hidden_size": h,
            "lr":          lr,
            "seed":        seed,
            "val_rmse":    val_rmse,
        })

val_df = pd.DataFrame(records)
val_df.to_csv(SRC_DIR / "results" / f"{prefix}lstm_study_val.csv", index=False)

# =============================================================================
# Select best config per approach (lowest mean val RMSE across seeds)
# =============================================================================
print("\n" + "=" * 72)
print("Validation summary (mean val RMSE across seeds)")
print("=" * 72)

for approach in APPROACHES:
    sub = val_df[val_df["approach"] == approach]
    agg = (
        sub.groupby("config_id")["val_rmse"]
        .agg(mean="mean", std="std")
        .sort_values("mean")
    )
    print(f"\n{approach}:")
    print(agg.to_string())

# =============================================================================
# Test evaluation of best config per approach
# =============================================================================
print("\n" + "=" * 72)
print("Test results for best configuration per approach")
print("=" * 72)

test_records = []
for approach in APPROACHES:
    sub     = val_df[val_df["approach"] == approach]
    best_id = sub.groupby("config_id")["val_rmse"].mean().idxmin()
    best    = sub[sub["config_id"] == best_id].iloc[0]
    w, h, lr = int(best["window_size"]), int(best["hidden_size"]), float(best["lr"])

    print(f"\n{approach} — best config: {best_id}"
          f"  (val RMSE {sub[sub['config_id']==best_id]['val_rmse'].mean():.4f}"
          f" ± {sub[sub['config_id']==best_id]['val_rmse'].std():.4f})")

    rmses, scores = [], []
    for seed in STUDY_SEEDS:
        cpath = ckpt_dir / f"{prefix}{ckpt_name(approach, w, h, lr, seed)}"
        rmse, score = _eval_mod.evaluate(approach, "lstm", seed=seed, ckpt_path=cpath, dataset=DATASET)
        rmses.append(rmse)
        scores.append(score)
        test_records.append({
            "approach": approach, "config_id": best_id,
            "seed": seed, "rmse": rmse, "nasa": score,
        })

    print(f"  Test RMSE:  {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"  NASA score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

pd.DataFrame(test_records).to_csv(
    SRC_DIR / "results" / f"{prefix}lstm_study_test.csv", index=False
)

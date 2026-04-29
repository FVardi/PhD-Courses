"""Aggregate multi-seed results and report mean ± std per configuration.

Reads src/results/all_results.csv produced by 1.9_run_all.py.
Reports a summary table and flags pairs whose difference is smaller than
one standard deviation in both metrics (not significant given the seed budget).

Usage:
    python 1.10_aggregate_results.py
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

SRC_DIR = Path(__file__).parents[1] / "src"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="FD001")
args, _ = parser.parse_known_args()

df = pd.read_csv(SRC_DIR / "results" / f"all_results_{args.dataset}.csv")

# --- summary table ---
summary = (
    df.groupby(["approach", "model"])
    .agg(
        rmse_mean=("rmse", "mean"),
        rmse_std= ("rmse", "std"),
        nasa_mean=("nasa", "mean"),
        nasa_std= ("nasa", "std"),
        n_seeds=  ("seed", "count"),
    )
    .reset_index()
    .sort_values("rmse_mean")
)

print("=" * 72)
print(f"{'Config':<22} {'RMSE (mean±std)':<22} {'NASA (mean±std)':<22}  seeds")
print("=" * 72)
for _, row in summary.iterrows():
    config = f"{row['approach']}/{row['model']}"
    rmse_s = f"{row['rmse_mean']:.4f} ± {row['rmse_std']:.4f}"
    nasa_s = f"{row['nasa_mean']:.2f} ± {row['nasa_std']:.2f}"
    print(f"{config:<22} {rmse_s:<22} {nasa_s:<22}  {int(row['n_seeds'])}")

# --- pairwise significance check ---
print("\n" + "=" * 72)
print("Pairwise significance (|mean_A − mean_B| < max(std_A, std_B) in BOTH metrics)")
print("=" * 72)

configs = summary.set_index(["approach", "model"])
keys    = list(configs.index)

not_significant = []
for (a1, m1), (a2, m2) in combinations(keys, 2):
    r1, r2 = configs.loc[(a1, m1)], configs.loc[(a2, m2)]

    rmse_diff  = abs(r1["rmse_mean"] - r2["rmse_mean"])
    rmse_thr   = max(r1["rmse_std"],  r2["rmse_std"])
    nasa_diff  = abs(r1["nasa_mean"]  - r2["nasa_mean"])
    nasa_thr   = max(r1["nasa_std"],  r2["nasa_std"])

    if rmse_diff < rmse_thr and nasa_diff < nasa_thr:
        not_significant.append(
            f"  {a1}/{m1}  vs  {a2}/{m2}"
            f"  (ΔRMSE={rmse_diff:.4f} < {rmse_thr:.4f},"
            f"  ΔNASA={nasa_diff:.2f} < {nasa_thr:.2f})"
        )

if not_significant:
    print("The following pairs are NOT significantly different given the seed budget:")
    for line in not_significant:
        print(line)
else:
    print("All pairwise differences exceed one standard deviation in at least one metric.")

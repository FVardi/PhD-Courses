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

# --- pairwise seed consistency check ---
print("\n" + "=" * 72)
print("Seed consistency check (within each approach/model combination)")
print("Flags seed pairs where |seed_A - seed_B| > std of that configuration")
print("=" * 72)

outliers = []
for (approach, model), group in df.groupby(["approach", "model"]):
    if len(group) < 2:
        continue
    rmse_std = group["rmse"].std()
    nasa_std = group["nasa"].std()

    for (i1, row1), (i2, row2) in combinations(group.iterrows(), 2):
        rmse_diff = abs(row1["rmse"] - row2["rmse"])
        nasa_diff = abs(row1["nasa"] - row2["nasa"])

        if rmse_diff > rmse_std or nasa_diff > nasa_std:
            outliers.append(
                f"  {approach}/{model}  seed {int(row1['seed'])} vs seed {int(row2['seed'])}"
                f"  (dRMSE={rmse_diff:.4f} > {rmse_std:.4f},"
                f"  dNASA={nasa_diff:.2f} > {nasa_std:.2f})"
            )

if outliers:
    print("The following seed pairs deviate by more than one std within their configuration:")
    for line in outliers:
        print(line)
else:
    print("All seed pairs within each configuration are consistent (differences ≤ one std).")

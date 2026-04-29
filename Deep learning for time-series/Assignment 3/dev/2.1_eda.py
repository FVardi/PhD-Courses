# %%
"""
2.1  EDA — FD002 (Multi-Condition, Single Fault Mode)
Explores the raw training data to motivate design choices in 2.2.
Nothing is modified or saved here — all analysis is on the raw split.
"""
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SRC_DIR     = Path(__file__).parents[1] / "src"
DATA_DIR    = SRC_DIR / "data" / "CMAPSSData"
RESULTS_DIR = SRC_DIR / "results"
FIGS_DIR    = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

with open(SRC_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

N_SAMPLE_ENGINES = cfg["eda"]["n_sample_engines"]
RANDOM_SEED      = cfg["split_seed"]
VAL_FRACTION     = cfg["val_fraction"]

DATASET     = "FD002"
COLUMNS     = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS     = ["op_setting_1", "op_setting_2", "op_setting_3"]

# %%  load raw training data — engine-wise train/val split (EDA on train only)
all_train = pd.read_csv(
    DATA_DIR / f"train_{DATASET}.txt",
    sep=r"\s+",
    header=None,
    names=COLUMNS,
    index_col=False,
)

units     = all_train["unit"].unique()
val_units = pd.Series(units).sample(frac=VAL_FRACTION, random_state=RANDOM_SEED)
train     = all_train[~all_train["unit"].isin(val_units)].reset_index(drop=True)
val       = all_train[ all_train["unit"].isin(val_units)].reset_index(drop=True)

print(f"Dataset: {DATASET}")
print(f"Train — engines: {train['unit'].nunique()}  rows: {len(train)}")
print(f"Val   — engines: {val['unit'].nunique()}  rows: {len(val)}")

# %%  operating-condition distributions (1-D histograms)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(f"{DATASET} — Operational setting distributions (train split)")
for ax, col in zip(axes, OP_COLS):
    ax.hist(train[col], bins=30, edgecolor="none")
    ax.set_title(col)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(FIGS_DIR / f"{DATASET}_op_settings.png", dpi=150)
plt.show()

# %%  2-D scatter coloured by op_setting_3
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(
    train["op_setting_1"], train["op_setting_2"],
    c=train["op_setting_3"], cmap="tab10", s=100, alpha=0.5,
)
plt.colorbar(sc, ax=ax, label="op_setting_3")
ax.set_xlabel("op_setting_1")
ax.set_ylabel("op_setting_2")
ax.set_title(f"{DATASET} — Operating conditions (coloured by op_setting_3)")
plt.tight_layout()
plt.savefig(FIGS_DIR / f"{DATASET}_op_conditions_scatter.png", dpi=150)
plt.show()

# %%  3-D scatter
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection="3d")
ax.scatter(
    train["op_setting_1"], train["op_setting_2"], train["op_setting_3"],
    s=2, alpha=0.4,
)
ax.set_xlabel("op_setting_1")
ax.set_ylabel("op_setting_2")
ax.set_zlabel("op_setting_3")
ax.set_title(f"{DATASET} — 3-D operating conditions")
plt.tight_layout()
plt.savefig(FIGS_DIR / f"{DATASET}_op_conditions_3d_scatter.png", dpi=150)
plt.show()

# %%  3-D histogram — how many of 20^3 bins are occupied?
N_BINS    = 20
op_data   = train[OP_COLS].values
counts, _ = np.histogramdd(op_data, bins=N_BINS)
print(f"Non-empty bins: {(counts > 0).sum()} / {counts.size} total")

# %%  sensor trajectories on train split
sample_units = train["unit"].unique()[:N_SAMPLE_ENGINES]
n_cols = 3
n_rows = -(-len(SENSOR_COLS) // n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3), sharex=False)
axes = axes.flatten()
for ax, sensor in zip(axes, SENSOR_COLS):
    for unit in sample_units:
        g = train[train["unit"] == unit]
        ax.plot(g["cycle"], g[sensor], alpha=0.6, linewidth=0.8)
    ax.set_title(sensor)
    ax.set_xlabel("cycle")
for ax in axes[len(SENSOR_COLS):]:
    ax.set_visible(False)
fig.suptitle(
    f"Sensor trajectories for {N_SAMPLE_ENGINES} engines — {DATASET} train split",
    y=1.01,
)
plt.tight_layout()
plt.savefig(FIGS_DIR / f"{DATASET}_sensor_trajectories.png", dpi=150, bbox_inches="tight")
plt.show()

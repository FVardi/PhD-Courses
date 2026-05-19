# %%
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

N_SAMPLE_ENGINES        = cfg["eda"]["n_sample_engines"]
LIFETIME_HIST_BINS      = cfg["eda"]["lifetime_hist_bins"]
SPEARMAN_MEAN_THRESH    = cfg["eda"]["spearman_mean_threshold"]
SPEARMAN_STD_THRESH     = cfg["eda"]["spearman_std_threshold"]
DATASET                 = cfg["dataset"]

COLUMNS = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

# %%
train = pd.read_csv(
    DATA_DIR / f"train_{DATASET}.txt",
    sep=r"\s+",
    header=None,
    names=COLUMNS,
    index_col=False,
)
print(f"Train shape: {train.shape}")
train.head()

# %%  lifetime distribution
lifetimes = train.groupby("unit")["cycle"].max().sort_values()

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(lifetimes, bins=LIFETIME_HIST_BINS, edgecolor="black")
ax.set_xlabel("Lifetime (cycles)")
ax.set_ylabel("Number of engines")
ax.set_title(f"Distribution of engine lifetimes – {DATASET} train set")
plt.tight_layout()
fig.savefig(FIGS_DIR / "lifetime_distribution.png", dpi=150)
plt.show()

# %%  raw sensor trajectories
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

fig.suptitle(f"Sensor trajectories for {N_SAMPLE_ENGINES} engines – {DATASET} train set", y=1.01)
plt.tight_layout()
fig.savefig(FIGS_DIR / "sensor_trajectories.png", dpi=150)
plt.show()

# %%  per-engine Spearman / Pearson correlation with cycle
spearman_raw = (
    train.groupby("unit")
    .apply(lambda g: g[SENSOR_COLS].corrwith(g["cycle"], method="spearman"))
)

corr = pd.DataFrame({
    "spearman_mean": spearman_raw.mean(),
    "spearman_std":  spearman_raw.std(),
}).sort_values("spearman_mean", key=abs, ascending=False)

print(corr.to_string(float_format="{:.3f}".format))

# %%  correlation bar chart
fig, ax = plt.subplots(figsize=(8, 6))

ordered = corr.sort_values("spearman_mean")
colors = ["steelblue" if v >= 0 else "tomato" for v in ordered["spearman_mean"]]
ax.barh(ordered.index, ordered["spearman_mean"], xerr=ordered["spearman_std"],
        color=colors, edgecolor="black", linewidth=0.4,
        error_kw={"elinewidth": 0.8, "ecolor": "black"})
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlim(-1, 1)
ax.set_xlabel("Mean Spearman ρ with cycle (± std across engines)")
ax.set_title(f"Per-engine sensor correlation with cycle – {DATASET} train set")
plt.tight_layout()
fig.savefig(FIGS_DIR / "sensor_correlations.png", dpi=150)
plt.show()

# %%  feature selection and export
mask = (
    (corr["spearman_mean"].abs() >= SPEARMAN_MEAN_THRESH) &
    (corr["spearman_std"]        <= SPEARMAN_STD_THRESH)
)
selected = corr[mask].index.tolist()
print(f"Selected {len(selected)} sensors:")
print(f"  |spearman_mean| >= {SPEARMAN_MEAN_THRESH}")
print(f"  spearman_std    <= {SPEARMAN_STD_THRESH}")
print(selected)

out = {
    "selected_sensors":        selected,
    "spearman_mean_threshold": SPEARMAN_MEAN_THRESH,
    "spearman_std_threshold":  SPEARMAN_STD_THRESH,
}
with open(RESULTS_DIR / "selected_features.yaml", "w") as f:
    yaml.dump(out, f)
print(f"Saved to {RESULTS_DIR / 'selected_features.yaml'}")


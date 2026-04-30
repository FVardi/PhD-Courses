# %%
"""
2.2  Data preparation — FD002 (Multi-Condition, Single Fault Mode)
Produces normalised train / val / test splits ready for model training.
Outputs saved to src/results/splits_FD002/.
"""
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

SRC_DIR     = Path(__file__).parents[1] / "src"
DATA_DIR    = SRC_DIR / "data" / "CMAPSSData"
RESULTS_DIR = SRC_DIR / "results"

with open(SRC_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

RUL_CAP      = cfg["rul_cap"]
RANDOM_SEED  = cfg["split_seed"]
VAL_FRACTION = cfg["val_fraction"]

DATASET     = "FD002"
COLUMNS     = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
OP_COLS     = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

# =============================================================================
# TRAIN / VAL  (derived from train_FD002.txt)
# =============================================================================

# %%  load raw train data and construct RUL labels
all_train = pd.read_csv(
    DATA_DIR / f"train_{DATASET}.txt",
    sep=r"\s+",
    header=None,
    names=COLUMNS,
    index_col=False,
)
all_train["RUL"] = (
    all_train.groupby("unit")["cycle"].transform("max") - all_train["cycle"]
).clip(upper=RUL_CAP)

print(f"Dataset: {DATASET}")
print(f"All train — engines: {all_train['unit'].nunique()}  rows: {len(all_train)}")

# %%  engine-wise train/val split
units     = all_train["unit"].unique()
val_units = pd.Series(units).sample(frac=VAL_FRACTION, random_state=RANDOM_SEED)
train = all_train[~all_train["unit"].isin(val_units)].reset_index(drop=True)
val   = all_train[ all_train["unit"].isin(val_units)].reset_index(drop=True)

print(f"Train — engines: {train['unit'].nunique()}  rows: {len(train)}")
print(f"Val   — engines: {val['unit'].nunique()}  rows: {len(val)}")

# %%  k-means clustering on operating conditions (fit on train only)
kmeans = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
train["op_condition"] = kmeans.fit_predict(train[OP_COLS])
print(f"Train cluster sizes:\n{train['op_condition'].value_counts().sort_index()}")

# %%  per-cluster z-score normalisation (statistics from train, applied to train and val)
cluster_stats = (
    train.groupby("op_condition")[SENSOR_COLS]
    .agg(["mean", "std"])
)
cluster_stats.columns = ["_".join(c) for c in cluster_stats.columns]
std_cols = [c for c in cluster_stats.columns if c.endswith("_std")]
cluster_stats[std_cols] = cluster_stats[std_cols].replace(0, 1)


def normalise_by_cluster(df, stats):
    out = df.copy()
    out[SENSOR_COLS] = out[SENSOR_COLS].astype(float)
    for cluster in df["op_condition"].unique():
        mask = df["op_condition"] == cluster
        for sensor in SENSOR_COLS:
            mean = stats.loc[cluster, f"{sensor}_mean"]
            std  = stats.loc[cluster, f"{sensor}_std"]
            std  = std if std > 0 else 1.0
            out.loc[mask, sensor] = (df.loc[mask, sensor] - mean) / std
    return out


train = normalise_by_cluster(train, cluster_stats)
print("Per-cluster normalisation applied to train.")

val["op_condition"] = kmeans.predict(val[OP_COLS])
val = normalise_by_cluster(val, cluster_stats)
print("Per-cluster normalisation applied to val.")
print(f"Val cluster sizes:\n{val['op_condition'].value_counts().sort_index()}")

# =============================================================================
# TEST  (derived from test_FD002.txt + RUL_FD002.txt)
# =============================================================================

# %%  load raw test data and construct RUL labels
rul_true = pd.read_csv(
    DATA_DIR / f"RUL_{DATASET}.txt",
    header=None,
    names=["RUL"],
)
test = pd.read_csv(
    DATA_DIR / f"test_{DATASET}.txt",
    sep=r"\s+",
    header=None,
    names=COLUMNS,
    index_col=False,
)
last_cycle = test.groupby("unit")["cycle"].max()
rul_map    = dict(enumerate(rul_true["RUL"], start=1))  # unit is 1-indexed
test["RUL"] = (
    test["unit"].map(last_cycle) - test["cycle"] + test["unit"].map(rul_map)
).clip(upper=RUL_CAP)

print(f"Test  — engines: {test['unit'].nunique()}  rows: {len(test)}")

test["op_condition"] = kmeans.predict(test[OP_COLS])
test = normalise_by_cluster(test, cluster_stats)
print("Per-cluster normalisation applied to test.")
print(f"Test cluster sizes:\n{test['op_condition'].value_counts().sort_index()}")

# =============================================================================
# SAVE
# =============================================================================

# %%
SPLITS_DIR = RESULTS_DIR / "splits_FD002"
SPLITS_DIR.mkdir(exist_ok=True)

train.to_parquet(SPLITS_DIR / "train.parquet", index=False)
val.to_parquet(  SPLITS_DIR / "val.parquet",   index=False)
test.to_parquet( SPLITS_DIR / "test.parquet",  index=False)

joblib.dump(kmeans, SPLITS_DIR / "kmeans.pkl")
cluster_stats.to_csv(SPLITS_DIR / "cluster_stats.csv")

print(f"Saved train / val / test, kmeans model, and cluster stats to {SPLITS_DIR}")

# =============================================================================
# VISUALISATION — raw vs normalised sensor trajectories
# =============================================================================

# %%
import matplotlib.pyplot as plt

FIGS_DIR = RESULTS_DIR / "figures"
FIGS_DIR.mkdir(exist_ok=True)

# reload raw training data
raw_train = pd.read_csv(
    DATA_DIR / f"train_{DATASET}.txt",
    sep=r"\s+", header=None, names=COLUMNS, index_col=False,
)

N_SAMPLE     = cfg["eda"]["n_sample_engines"]
sample_units = train["unit"].unique()[:N_SAMPLE]

# %%  one figure per sample engine — raw (left) vs normalised (right)
for unit in sample_units:
    raw_eng  = raw_train[raw_train["unit"] == unit].sort_values("cycle")
    norm_eng = train[train["unit"] == unit].sort_values("cycle")
    cycles   = raw_eng["cycle"].values

    fig, axes = plt.subplots(
        len(SENSOR_COLS), 2,
        figsize=(12, 1.8 * len(SENSOR_COLS)),
        sharex=True,
    )
    axes[0, 0].set_title("Raw")
    axes[0, 1].set_title("Normalised")

    for i, sensor in enumerate(SENSOR_COLS):
        axes[i, 0].plot(cycles, raw_eng[sensor].values,  linewidth=0.8)
        axes[i, 1].plot(cycles, norm_eng[sensor].values, linewidth=0.8)
        axes[i, 1].axhline(0, color="k", linewidth=0.4, linestyle="--")
        for col in (0, 1):
            axes[i, col].set_ylabel(sensor, fontsize=7)

    axes[-1, 0].set_xlabel("cycle")
    axes[-1, 1].set_xlabel("cycle")

    fig.suptitle(f"Engine {unit} — raw vs normalised sensors (FD002)", y=1.002)
    plt.tight_layout()
    plt.savefig(
        FIGS_DIR / f"FD002_engine{unit}_raw_vs_norm.png",
        dpi=150, bbox_inches="tight",
    )
    plt.show()

# %%
import yaml
import pandas as pd
from pathlib import Path

SRC_DIR     = Path(__file__).parents[1] / "src"
DATA_DIR    = SRC_DIR / "data" / "CMAPSSData"
RESULTS_DIR = SRC_DIR / "results"

with open(SRC_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)

DATASET      = cfg["dataset"]
RUL_CAP      = cfg["rul_cap"]
RANDOM_SEED  = cfg["random_seed"]
VAL_FRACTION = cfg["val_fraction"]

COLUMNS = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

with open(RESULTS_DIR / "selected_features.yaml") as f:
    features = yaml.safe_load(f)

SELECTED_SENSORS = features["selected_sensors"]

# =============================================================================
# TRAIN / VALIDATION  (derived from train_FD001.txt)
# =============================================================================

# %%  load raw train data
all_train = pd.read_csv(
    DATA_DIR / f"train_{DATASET}.txt",
    sep=r"\s+",
    header=None,
    names=COLUMNS,
    index_col=False,
)
all_train["RUL"] = (
    (all_train.groupby("unit")["cycle"].transform("max") - all_train["cycle"])
    .clip(upper=RUL_CAP)
)
print(f"All train — engines: {all_train['unit'].nunique()}  rows: {len(all_train)}")

# %%  engine-wise split into train and validation
units     = all_train["unit"].unique()
val_units = pd.Series(units).sample(frac=VAL_FRACTION, random_state=RANDOM_SEED)

train = all_train[~all_train["unit"].isin(val_units)].reset_index(drop=True)
val   = all_train[ all_train["unit"].isin(val_units)].reset_index(drop=True)

print(f"Train — engines: {train['unit'].nunique()}  rows: {len(train)}")
print(f"Val   — engines: {val['unit'].nunique()}  rows: {len(val)}")

# %%  fit normalisation statistics on train set only (per sensor channel)
sensor_mean = train[SELECTED_SENSORS].mean()
sensor_std  = train[SELECTED_SENSORS].std().replace(0, 1)  # avoid division by zero

def normalise(df):
    out = df.copy()
    out[SELECTED_SENSORS] = (df[SELECTED_SENSORS] - sensor_mean) / sensor_std
    return out

train = normalise(train)
val   = normalise(val)

# =============================================================================
# TEST  (derived from test_FD001.txt + RUL_FD001.txt, loaded after train/val)
# =============================================================================

# %%  load raw test data
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
test["RUL"] = (
    test["unit"].map(last_cycle) - test["cycle"] + test["unit"].map(rul_true["RUL"])
).clip(upper=RUL_CAP)

test = normalise(test)
print(f"Test  — engines: {test['unit'].nunique()}  rows: {len(test)}")

# =============================================================================
# SAVE
# =============================================================================

# %%
SPLITS_DIR = RESULTS_DIR / "splits"
SPLITS_DIR.mkdir(exist_ok=True)

train.to_parquet(SPLITS_DIR / "train.parquet", index=False)
val.to_parquet(  SPLITS_DIR / "val.parquet",   index=False)
test.to_parquet( SPLITS_DIR / "test.parquet",  index=False)

normalisation = pd.DataFrame({"mean": sensor_mean, "std": sensor_std})
normalisation.to_csv(SPLITS_DIR / "normalisation.csv")

print(f"Saved train / val / test and normalisation stats to {SPLITS_DIR}")

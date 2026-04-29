"""XGBoost baseline for RUL prediction.

Reuses FeatureSequenceDataset for feature extraction.
Each window position is treated as an independent sample (no sequential structure).

Usage (CLI):
    python 1.8_xgboost_baseline.py --seed 42

Usage (cells): set SEED below, then run all cells.
"""

# %%
import argparse
import importlib.util
import random
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

SRC_DIR = Path(__file__).parents[1] / "src"
DEV_DIR = Path(__file__).parent


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, DEV_DIR / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_datasets = _load_module("datasets", "1.3_datasets.py")
FeatureSequenceDataset = _datasets.FeatureSequenceDataset


def dataset_to_arrays(ds):
    X = np.concatenate([seq.numpy() for seq, _ in ds], axis=0)
    y = np.concatenate([rul.numpy() for _, rul in ds], axis=0)
    return X, y


def nasa_score(preds, targets):
    d = preds - targets
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(s.sum())


def _dataset_cfg(dataset):
    results_dir = SRC_DIR / "results"
    if dataset == "FD001":
        splits_dir = results_dir / "splits"
        with open(results_dir / "selected_features.yaml") as f:
            sensor_cols = yaml.safe_load(f)["selected_sensors"]
    else:
        splits_dir  = results_dir / f"splits_{dataset}"
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    return splits_dir, sensor_cols


def run(seed=42, dataset="FD001"):
    random.seed(seed)
    np.random.seed(seed)

    with open(SRC_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    splits_dir, selected_sensors = _dataset_cfg(dataset)
    prefix      = "" if dataset == "FD001" else f"{dataset}_"
    window_size = cfg["window_size"]

    train_ds = FeatureSequenceDataset(splits_dir / "train.parquet", selected_sensors, window_size)
    val_ds   = FeatureSequenceDataset(splits_dir / "val.parquet",   selected_sensors, window_size)
    test_ds  = FeatureSequenceDataset(splits_dir / "test.parquet",  selected_sensors, window_size)

    X_train, y_train = dataset_to_arrays(train_ds)
    X_val,   y_val   = dataset_to_arrays(val_ds)
    X_test,  y_test  = dataset_to_arrays(test_ds)

    print(f"Dataset: {dataset}  |  Seed: {seed}  |  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    best_params_path = SRC_DIR / "results" / f"{prefix}xgboost_best_params.yaml"
    if best_params_path.exists():
        with open(best_params_path) as f:
            tuned = yaml.safe_load(f)
        print(f"Loaded tuned params from {best_params_path}")
    else:
        tuned = {}
        print("No tuned params found — using defaults")

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=tuned.get(    "learning_rate",    0.05),
        max_depth=tuned.get(        "max_depth",        6),
        subsample=tuned.get(        "subsample",        0.8),
        colsample_bytree=tuned.get( "colsample_bytree", 0.8),
        min_child_weight=tuned.get( "min_child_weight", 1),
        reg_lambda=tuned.get(       "reg_lambda",       1.0),
        early_stopping_rounds=30,
        eval_metric="rmse",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    preds         = model.predict(X_test)
    final_preds   = preds[test_ds.last_indices]
    final_targets = y_test[test_ds.last_indices]

    rmse  = float(np.sqrt(((preds - y_test) ** 2).mean()))
    score = nasa_score(final_preds, final_targets)
    print(f"\nTest RMSE:  {rmse:.4f}")
    print(f"NASA score: {score:.2f}")

    results_dir = SRC_DIR / "results" / "predictions"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{prefix}sequence_xgboost_seed{seed}.parquet"
    pd.DataFrame({"pred": preds, "target": y_test}).to_parquet(out_path, index=False)
    print(f"Predictions saved to {out_path}")

    return rmse, score


# %%  config — edit when running as cells
SEED = 42

# %%  run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args, _ = parser.parse_known_args()
    run(args.seed)

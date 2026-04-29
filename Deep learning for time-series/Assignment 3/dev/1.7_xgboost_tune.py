"""Hyperparameter tuning for XGBoost RUL baseline using Optuna.

Searches over key XGBoost hyperparameters; early stopping handles n_estimators.
Best parameters are saved and can be pasted into 1.7_xgboost_baseline.py.

Usage (CLI):
    python 1.8_xgboost_tune.py --trials 100
    python 1.8_xgboost_tune.py --trials 50 --jobs 4

Usage (cells): set N_TRIALS below, then run all cells.
"""

# %%
import argparse
import importlib.util
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import optuna
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


# %%  load data
with open(SRC_DIR / "config.yaml") as f:
    cfg = yaml.safe_load(f)
with open(SRC_DIR / "results" / "selected_features.yaml") as f:
    selected_sensors = yaml.safe_load(f)["selected_sensors"]

splits_dir  = SRC_DIR / "results" / "splits"
window_size = cfg["window_size"]

train_ds = FeatureSequenceDataset(splits_dir / "train.parquet", selected_sensors, window_size)
val_ds   = FeatureSequenceDataset(splits_dir / "val.parquet",   selected_sensors, window_size)

X_train, y_train = dataset_to_arrays(train_ds)
X_val,   y_val   = dataset_to_arrays(val_ds)

print(f"Train: {X_train.shape}  Val: {X_val.shape}")


# %%  objective
def objective(trial):
    params = {
        "n_estimators":      1000,   # controlled by early stopping
        "learning_rate":     trial.suggest_float("learning_rate",    1e-3, 0.3,  log=True),
        "max_depth":         trial.suggest_int(  "max_depth",        3,    10),
        "subsample":         trial.suggest_float("subsample",        0.5,  1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5,  1.0),
        "min_child_weight":  trial.suggest_int(  "min_child_weight", 1,    20),
        "reg_lambda":        trial.suggest_float("reg_lambda",       1e-2, 10.0, log=True),
        "early_stopping_rounds": 30,
        "eval_metric": "rmse",
        "random_state": cfg["split_seed"],
        "n_jobs": -1,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds    = model.predict(X_val)
    val_rmse = float(np.sqrt(((preds - y_val) ** 2).mean()))
    return val_rmse


# %%  run study
N_TRIALS = 100   # edit when running as cells

def tune(n_trials):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=cfg["split_seed"])
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest val RMSE: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out_path = SRC_DIR / "results" / "xgboost_best_params.yaml"
    with open(out_path, "w") as f:
        yaml.dump(study.best_params, f, default_flow_style=False)
    print(f"\nSaved to {out_path}")

    return study


# %%  run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args, _ = parser.parse_known_args()
    study = tune(args.trials)

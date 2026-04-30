"""Evaluate a trained RUL model on the test set.

Usage (CLI):
    python 1.6_evaluate.py --approach window   --model lstm
    python 1.6_evaluate.py --approach sequence --model tcn

Usage (cells): set APPROACH and MODEL in the config cell, then run all cells.
"""

# %%
import argparse
import importlib.util
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

SRC_DIR = Path(__file__).parents[1] / "src"
DEV_DIR = Path(__file__).parent


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, DEV_DIR / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_datasets = _load_module("datasets", "1.3_datasets.py")
_models   = _load_module("models",   "1.4_models.py")

SlidingWindowDataset   = _datasets.SlidingWindowDataset
FeatureSequenceDataset = _datasets.FeatureSequenceDataset
sequence_collate_fn    = _datasets.sequence_collate_fn
get_model              = _models.get_model


def nasa_score(preds, targets):
    d = preds - targets
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(s.sum())


def predict(model, loader, approach, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            if approach == "window":
                x, y = batch
                pred = model(x.to(device))[:, -1, 0].cpu()    # (batch,)
                preds.append(pred)
                targets.append(y)
            else:
                x, y, lengths = batch
                pred = model(x.to(device), lengths)[:, :, 0].cpu()
                idx  = lengths - 1
                preds.append(pred[torch.arange(len(idx)), idx])
                targets.append(y[torch.arange(len(idx)), idx])

    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def _dataset_cfg(dataset):
    results_dir = SRC_DIR / "results"
    splits_dir  = results_dir / f"splits_{dataset}"
    with open(results_dir / "selected_features.yaml") as f:
        sensor_cols = yaml.safe_load(f)["selected_sensors"]
    extra_cols = ["op_condition"] if dataset != "FD001" else []
    return splits_dir, sensor_cols, extra_cols


def evaluate(approach, model_name, seed=42, ckpt_path=None, dataset="FD001"):
    prefix     = "" if dataset == "FD001" else f"{dataset}_"
    splits_dir, selected_sensors, extra_cols = _dataset_cfg(dataset)
    if ckpt_path is None:
        ckpt_path = SRC_DIR / "results" / "checkpoints" / f"{prefix}{approach}_{model_name}_seed{seed}.pt"

    ckpt   = torch.load(ckpt_path, map_location="cpu")
    cfg    = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    window_size = cfg["window_size"]
    batch_size  = cfg["training"]["batch_size"]

    if approach == "window":
        test_ds = SlidingWindowDataset(splits_dir / "test.parquet", selected_sensors, window_size, extra_cols)
        loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        test_ds = FeatureSequenceDataset(splits_dir / "test.parquet", selected_sensors, window_size, extra_cols)
        loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=sequence_collate_fn, num_workers=0)

    model = get_model(model_name, ckpt["input_size"], cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    preds, targets = predict(model, loader, approach, device)

    # per-engine final predictions for NASA score
    if approach == "window":
        final_preds   = preds[test_ds.last_indices]
        final_targets = targets[test_ds.last_indices]
    else:
        # sequence approach already returns one prediction per engine
        final_preds, final_targets = preds, targets

    rmse  = float(np.sqrt(((final_preds - final_targets) ** 2).mean()))
    score = nasa_score(final_preds, final_targets)
    print(f"Approach: {approach} | Model: {model_name}")
    print(f"  Test RMSE:       {rmse:.4f}")
    print(f"  NASA score:      {score:.2f}")

    results_dir = SRC_DIR / "results" / "predictions"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{prefix}{approach}_{model_name}_seed{seed}.parquet"
    pd.DataFrame({"pred": preds, "target": targets}).to_parquet(out_path, index=False)
    print(f"  Predictions saved to {out_path}")
    return rmse, score


# %%  config — edit these when running as cells
APPROACH = "window"   # "window" or "sequence"
MODEL    = "lstm"     # "rnn", "lstm", or "tcn"
SEED     = 42

# %%  run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=["window", "sequence"], default=APPROACH)
    parser.add_argument("--model",    choices=["rnn", "lstm", "tcn"],  default=MODEL)
    parser.add_argument("--seed",     type=int,                        default=SEED)
    args, _ = parser.parse_known_args()
    evaluate(args.approach, args.model, args.seed)

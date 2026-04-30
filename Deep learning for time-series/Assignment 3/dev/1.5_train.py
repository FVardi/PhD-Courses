"""Train a RUL prediction model.

Usage (CLI):
    python 1.5_train.py --approach window   --model lstm
    python 1.5_train.py --approach sequence --model tcn

Usage (cells): set APPROACH and MODEL in the config cell, then run all cells.
"""

# %%
import argparse
import importlib.util
import random
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
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


def masked_mse(pred, target, lengths):
    mask = (
        torch.arange(pred.shape[1], device=pred.device)[None, :]
        < lengths[:, None].to(pred.device)
    ).float()
    return ((pred - target) ** 2 * mask).sum() / mask.sum()


def run_epoch(model, loader, approach, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_weight = 0.0, 0.0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if approach == "window":
                x, y = batch
                x, y  = x.to(device), y.to(device)
                pred   = model(x)[:, -1, 0]
                loss   = nn.functional.mse_loss(pred, y)
                weight = y.numel()
            else:
                x, y, lengths = batch
                x, y   = x.to(device), y.to(device)
                pred   = model(x, lengths)[:, :, 0]
                loss   = masked_mse(pred, y, lengths)
                weight = lengths.sum().item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss   += loss.item() * weight
            total_weight += weight

    return total_loss / total_weight


def _dataset_cfg(dataset, cfg):
    results_dir = SRC_DIR / "results"
    splits_dir  = results_dir / f"splits_{dataset}"
    with open(results_dir / "selected_features.yaml") as f:
        sensor_cols = yaml.safe_load(f)["selected_sensors"]
    extra_cols = ["op_condition"] if dataset != "FD001" else []
    return splits_dir, sensor_cols, extra_cols


def train(approach, model_name, seed=42, cfg=None, ckpt_name=None, dataset="FD001"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cfg is None:
        with open(SRC_DIR / "config.yaml") as f:
            cfg = yaml.safe_load(f)

    splits_dir, selected_sensors, extra_cols = _dataset_cfg(dataset, cfg)
    prefix   = "" if dataset == "FD001" else f"{dataset}_"
    ckpt_dir = SRC_DIR / "results" / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    window_size = cfg["window_size"]
    t_cfg       = cfg["training"]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Dataset: {dataset}  |  Approach: {approach}  |  Model: {model_name}  |  Seed: {seed}")

    if approach == "window":
        train_ds = SlidingWindowDataset(splits_dir / "train.parquet", selected_sensors, window_size, extra_cols)
        val_ds   = SlidingWindowDataset(splits_dir / "val.parquet",   selected_sensors, window_size, extra_cols)
        train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=t_cfg["batch_size"], shuffle=False, num_workers=0)
    else:
        train_ds = FeatureSequenceDataset(splits_dir / "train.parquet", selected_sensors, window_size, extra_cols)
        val_ds   = FeatureSequenceDataset(splits_dir / "val.parquet",   selected_sensors, window_size, extra_cols)
        train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
                                  collate_fn=sequence_collate_fn, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=t_cfg["batch_size"], shuffle=False,
                                  collate_fn=sequence_collate_fn, num_workers=0)

    input_size = train_ds[0][0].shape[-1]
    print(f"Input size: {input_size}  |  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    model     = get_model(model_name, input_size, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_rmse     = float("inf")
    epochs_no_improve = 0
    patience          = t_cfg["early_stopping_patience"]
    _default_name = f"{prefix}{approach}_{model_name}_seed{seed}.pt"
    ckpt_path     = ckpt_dir / (ckpt_name if ckpt_name else _default_name)

    for epoch in range(1, t_cfg["epochs"] + 1):
        train_mse = run_epoch(model, train_loader, approach, optimizer, device)
        val_mse   = run_epoch(model, val_loader,   approach, None,      device)
        val_rmse  = val_mse ** 0.5
        scheduler.step(val_rmse)

        print(f"Epoch {epoch:03d}  train RMSE: {train_mse**0.5:.4f}  val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse     = val_rmse
            epochs_no_improve = 0
            ckpt_data = {
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_rmse":    val_rmse,
                "approach":    approach,
                "model_name":  model_name,
                "input_size":  input_size,
                "cfg":         cfg,
            }
            for _attempt in range(5):
                try:
                    torch.save(ckpt_data, ckpt_path)
                    break
                except RuntimeError:
                    time.sleep(0.5)
            print(f"  -> checkpoint saved (val RMSE {val_rmse:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest val RMSE: {best_val_rmse:.4f}  checkpoint: {ckpt_path}")
    return best_val_rmse


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
    train(args.approach, args.model, args.seed)

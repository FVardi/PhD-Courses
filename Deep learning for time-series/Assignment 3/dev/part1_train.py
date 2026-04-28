"""Train a RUL prediction model.

Usage:
    python part2_train.py --approach window   --model lstm
    python part2_train.py --approach sequence --model tcn
"""

import argparse
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

SRC_DIR = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(Path(__file__).parent))

from part2_datasets import SlidingWindowDataset, FeatureSequenceDataset
from part2_models import get_model


def masked_mse(pred, target, mask):
    """MSE over non-padded positions only."""
    loss = (pred - target) ** 2 * mask
    return loss.sum() / mask.sum()


def run_epoch(model, loader, approach, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_weight = 0.0, 0.0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if approach == "window":
                x, y = batch
                x, y = x.to(device), y.to(device)
                pred = model(x)[:, -1, 0]              # (batch,)
                loss = nn.functional.mse_loss(pred, y)
                weight = y.numel()
            else:
                x, y, mask = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                pred = model(x)[:, :, 0]               # (batch, T)
                loss = masked_mse(pred, y, mask)
                weight = mask.sum().item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss   += loss.item() * weight
            total_weight += weight

    return total_loss / total_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=["window", "sequence"], required=True)
    parser.add_argument("--model",    choices=["rnn", "lstm", "tcn"],  required=True)
    args = parser.parse_args()

    with open(SRC_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    with open(SRC_DIR / "results" / "selected_features.yaml") as f:
        selected_sensors = yaml.safe_load(f)["selected_sensors"]

    splits_dir  = SRC_DIR / "results" / "splits"
    ckpt_dir    = SRC_DIR / "results" / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    window_size = cfg["window_size"]
    t_cfg       = cfg["training"]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    Dataset = SlidingWindowDataset if args.approach == "window" else FeatureSequenceDataset

    train_ds = Dataset(splits_dir / "train.parquet", selected_sensors, window_size)
    val_ds   = Dataset(splits_dir / "val.parquet",   selected_sensors, window_size)

    train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=t_cfg["batch_size"], shuffle=False, num_workers=0)

    input_size = train_ds[0][0].shape[-1]
    print(f"Approach: {args.approach} | Model: {args.model} | Input size: {input_size}")

    # -------------------------------------------------------------------------
    # Model, optimiser, scheduler
    # -------------------------------------------------------------------------
    model     = get_model(args.model, input_size, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    best_val_loss = float("inf")
    ckpt_path     = ckpt_dir / f"{args.approach}_{args.model}.pt"

    for epoch in range(1, t_cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, args.approach, optimizer, device)
        val_loss   = run_epoch(model, val_loader,   args.approach, None,      device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d}  train MSE: {train_loss:.4f}  val MSE: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "approach":    args.approach,
                "model_name":  args.model,
                "input_size":  input_size,
                "cfg":         cfg,
            }, ckpt_path)
            print(f"  → saved checkpoint (val MSE {val_loss:.4f})")

    print(f"\nBest val MSE: {best_val_loss:.4f}  checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

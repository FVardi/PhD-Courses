"""Evaluate a trained RUL model on the test set.

Usage:
    python part2_evaluate.py --approach window   --model lstm
    python part2_evaluate.py --approach sequence --model tcn
"""

import argparse
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

SRC_DIR = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(Path(__file__).parent))

from part2_datasets import SlidingWindowDataset, FeatureSequenceDataset
from part2_models import get_model


def predict(model, loader, approach, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            if approach == "window":
                x, y = batch
                x = x.to(device)
                pred = model(x)[:, -1, 0].cpu()            # (batch,)
                preds.append(pred)
                targets.append(y)
            else:
                x, y, mask = batch
                x, mask = x.to(device), mask.to(device)
                pred = model(x)[:, :, 0].cpu()             # (batch, T)
                # keep only valid (non-padded) positions
                m = mask.cpu().bool()
                preds.append(pred[m])
                targets.append(y[m])

    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=["window", "sequence"], required=True)
    parser.add_argument("--model",    choices=["rnn", "lstm", "tcn"],  required=True)
    args = parser.parse_args()

    splits_dir = SRC_DIR / "results" / "splits"
    ckpt_path  = SRC_DIR / "results" / "checkpoints" / f"{args.approach}_{args.model}.pt"

    ckpt   = torch.load(ckpt_path, map_location="cpu")
    cfg    = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(SRC_DIR / "results" / "selected_features.yaml") as f:
        selected_sensors = yaml.safe_load(f)["selected_sensors"]

    Dataset  = SlidingWindowDataset if args.approach == "window" else FeatureSequenceDataset
    test_ds  = Dataset(splits_dir / "test.parquet", selected_sensors, cfg["window_size"])
    loader   = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    model = get_model(args.model, ckpt["input_size"], cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    preds, targets = predict(model, loader, args.approach, device)

    rmse = float(np.sqrt(((preds - targets) ** 2).mean()))
    mae  = float(np.abs(preds - targets).mean())
    print(f"Approach: {args.approach} | Model: {args.model}")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  Test MAE:  {mae:.4f}")

    results_dir = SRC_DIR / "results" / "predictions"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{args.approach}_{args.model}.parquet"
    pd.DataFrame({"pred": preds, "target": targets}).to_parquet(out_path, index=False)
    print(f"  Predictions saved to {out_path}")


if __name__ == "__main__":
    main()

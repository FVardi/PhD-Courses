import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class SlidingWindowDataset(Dataset):
    """One sample = one fixed-size window of raw sensor data.
    Label = RUL at the last timestep of the window.
    Input shape: (window_size, n_sensors).
    """

    def __init__(self, parquet_path, selected_sensors, window_size):
        df = pd.read_parquet(parquet_path)
        windows, labels = [], []

        for _, group in df.groupby("unit"):
            data = group[selected_sensors].values.astype(np.float32)  # (T, n_sensors)
            rul  = group["RUL"].values.astype(np.float32)             # (T,)
            T    = len(data)

            if T < window_size:
                continue

            n_valid = T - window_size + 1
            idx = np.arange(window_size)[None, :] + np.arange(n_valid)[:, None]  # (n_valid, W)
            windows.append(data[idx])                       # (n_valid, W, n_sensors)
            labels.append(rul[window_size - 1:])            # (n_valid,)

        self.windows = torch.tensor(np.concatenate(windows, axis=0))  # (N, W, n_sensors)
        self.labels  = torch.tensor(np.concatenate(labels,  axis=0))  # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


class FeatureSequenceDataset(Dataset):
    """One sample = one engine's full feature sequence (padded).
    Features per timestep: rolling mean, std, OLS slope, deviation from baseline
    — each computed over the window ending at that timestep.
    Input shape: (max_seq_len, n_sensors * 4).
    Label shape: (max_seq_len,) with binary mask.
    """

    def __init__(self, parquet_path, selected_sensors, window_size):
        df = pd.read_parquet(parquet_path)
        n_sensors = len(selected_sensors)
        n_features = n_sensors * 4

        # Pre-compute OLS terms (same for all windows)
        x     = np.arange(window_size, dtype=np.float32)
        x_c   = x - x.mean()
        x_var = (x_c ** 2).sum()

        all_features, all_rul = [], []

        for _, group in df.groupby("unit"):
            data = group[selected_sensors].values.astype(np.float32)  # (T, n_sensors)
            rul  = group["RUL"].values.astype(np.float32)             # (T,)
            T    = len(data)

            if T < window_size:
                continue

            n_valid = T - window_size + 1
            idx     = np.arange(window_size)[None, :] + np.arange(n_valid)[:, None]
            windows = data[idx]                             # (n_valid, W, n_sensors)

            rolling_mean = windows.mean(axis=1)             # (n_valid, n_sensors)
            rolling_std  = windows.std(axis=1)              # (n_valid, n_sensors)

            y_c   = windows - rolling_mean[:, None, :]      # centre each window
            slope = (x_c[None, :, None] * y_c).sum(axis=1) / x_var  # (n_valid, n_sensors)

            baseline_mean = rolling_mean[0]                 # (n_sensors,) — first window
            deviation     = rolling_mean - baseline_mean    # (n_valid, n_sensors)

            feats = np.concatenate(
                [rolling_mean, rolling_std, slope, deviation], axis=1
            )  # (n_valid, n_features)

            all_features.append(feats)
            all_rul.append(rul[window_size - 1:])           # (n_valid,)

        # Pad all sequences to global maximum length
        max_len = max(f.shape[0] for f in all_features)

        self.features = torch.zeros(len(all_features), max_len, n_features)
        self.rul      = torch.zeros(len(all_features), max_len)
        self.mask     = torch.zeros(len(all_features), max_len)

        for i, (feat, rul_seq) in enumerate(zip(all_features, all_rul)):
            L = feat.shape[0]
            self.features[i, :L] = torch.tensor(feat)
            self.rul[i, :L]      = torch.tensor(rul_seq)
            self.mask[i, :L]     = 1.0

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.rul[idx], self.mask[idx]

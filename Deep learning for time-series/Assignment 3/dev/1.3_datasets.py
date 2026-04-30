import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd


class SlidingWindowDataset(Dataset):
    """One sample = one fixed-size window of raw sensor data.
    Label = RUL at the last timestep of the window.
    Input shape: (window_size, n_sensors).
    """

    def __init__(self, parquet_path, selected_sensors, window_size, extra_cols=None):
        df = pd.read_parquet(parquet_path)
        extra_cols = extra_cols or []
        windows, labels = [], []

        for _, group in df.groupby("unit"):
            data = group[selected_sensors].values.astype(np.float32)  # (T, n_sensors)
            rul  = group["RUL"].values.astype(np.float32)             # (T,)
            T    = len(data)

            if T < window_size:
                continue

            n_valid = T - window_size + 1
            idx = np.arange(window_size)[None, :] + np.arange(n_valid)[:, None]
            win = data[idx]                                            # (n_valid, W, n_sensors)
            if extra_cols:
                extra = group[extra_cols].values.astype(np.float32)   # (T, n_extra)
                win   = np.concatenate([win, extra[idx]], axis=2)      # (n_valid, W, n_sensors+n_extra)
            windows.append(win)
            labels.append(rul[window_size - 1:])                      # (n_valid,)

        self.windows = torch.tensor(np.concatenate(windows, axis=0))  # (N, W, n_sensors)
        self.labels  = torch.tensor(np.concatenate(labels,  axis=0))  # (N,)
        # flat index of each engine's final window (for NASA score)
        sizes = [len(l) for l in labels]
        ends  = np.cumsum(sizes) - 1
        self.last_indices = ends.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


class FeatureSequenceDataset(Dataset):
    """One sample = one engine's full (unpadded) feature sequence.

    Features per timestep: rolling mean, std, OLS slope, deviation from baseline
    — each computed over the window ending at that timestep.
    Input shape: (seq_len, n_sensors * 4)  — variable length per engine.
    Label shape: (seq_len,)                — variable length per engine.

    Use sequence_collate_fn with the DataLoader to pad per-batch and get lengths.
    """

    def __init__(self, parquet_path, selected_sensors, window_size, extra_cols=None):
        df = pd.read_parquet(parquet_path)
        extra_cols = extra_cols or []
        x     = np.arange(window_size, dtype=np.float32)
        x_c   = x - x.mean()
        x_var = (x_c ** 2).sum()

        self.features = []
        self.rul      = []

        for _, group in df.groupby("unit"):
            data = group[selected_sensors].values.astype(np.float32)  # (T, n_sensors)
            rul  = group["RUL"].values.astype(np.float32)             # (T,)
            T    = len(data)

            if T < window_size:
                continue

            n_valid = T - window_size + 1
            idx     = np.arange(window_size)[None, :] + np.arange(n_valid)[:, None]
            windows = data[idx]                              # (n_valid, W, n_sensors)

            rolling_mean  = windows.mean(axis=1)            # (n_valid, n_sensors)
            rolling_std   = windows.std(axis=1)             # (n_valid, n_sensors)
            y_c           = windows - rolling_mean[:, None, :]
            slope         = (x_c[None, :, None] * y_c).sum(axis=1) / x_var
            deviation     = rolling_mean - rolling_mean[0]  # diff from first window

            feats = np.concatenate(
                [rolling_mean, rolling_std, slope, deviation], axis=1
            ).astype(np.float32)                            # (n_valid, n_sensors * 4)

            if extra_cols:
                extra = group[extra_cols].values.astype(np.float32)   # (T, n_extra)
                feats = np.concatenate([feats, extra[window_size - 1:]], axis=1)

            self.features.append(torch.tensor(feats))
            self.rul.append(torch.tensor(rul[window_size - 1:]))

        # flat index of each engine's final row (for NASA score)
        sizes = [f.shape[0] for f in self.features]
        ends  = np.cumsum(sizes) - 1
        self.last_indices = ends.tolist()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.rul[idx]


def sequence_collate_fn(batch):
    """Pad a batch of variable-length sequences and return real lengths.

    Sorts by descending length as required by pack_padded_sequence.

    Returns:
        padded_x:   (batch, T_max, n_features)
        padded_y:   (batch, T_max)
        lengths:    (batch,)  — real sequence length of each engine
    """
    features, ruls = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)

    order      = lengths.argsort(descending=True)
    features   = [features[i] for i in order]
    ruls       = [ruls[i]      for i in order]
    lengths    = lengths[order]

    padded_x = pad_sequence(features, batch_first=True)   # (batch, T_max, n_features)
    padded_y = pad_sequence(ruls,     batch_first=True)   # (batch, T_max)

    return padded_x, padded_y, lengths

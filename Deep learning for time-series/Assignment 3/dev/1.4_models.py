import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        out, _ = self.rnn(x)
        if lengths is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)
        return self.fc(out)     # (batch, T, 1)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        out, _ = self.lstm(x)
        if lengths is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)
        return self.fc(out)     # (batch, T, 1)


# ---------------------------------------------------------------------------
# TCN building blocks
# ---------------------------------------------------------------------------

class _CausalConv1d(nn.Module):
    """1D convolution with left-only (causal) padding."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
    """Two causal conv layers with a residual connection."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1      = _CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
        self.conv2      = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.dropout    = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)
        out = self.dropout(F.relu(self.conv1(x)))
        out = self.dropout(F.relu(self.conv2(out)))
        return F.relu(out + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network with exponentially growing dilations."""

    def __init__(self, input_size, num_channels, kernel_size, dropout=0.0):
        super().__init__()
        blocks = []
        in_ch  = input_size
        for i, out_ch in enumerate(num_channels):
            blocks.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*blocks)
        self.fc      = nn.Linear(num_channels[-1], 1)

    def forward(self, x, lengths=None):
        # lengths unused — TCN is not recurrent, padding does not affect valid positions
        out = self.network(x.transpose(1, 2))   # (batch, channels, T)
        out = out.transpose(1, 2)               # (batch, T, channels)
        return self.fc(out)                     # (batch, T, 1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(name: str, input_size: int, cfg: dict) -> nn.Module:
    t = cfg["training"]
    if name == "rnn":
        return RNNModel(input_size, t["hidden_size"], t["num_layers"], t["dropout"])
    if name == "lstm":
        return LSTMModel(input_size, t["hidden_size"], t["num_layers"], t["dropout"])
    if name == "tcn":
        c = cfg["tcn"]
        return TCNModel(input_size, c["num_channels"], c["kernel_size"], c["dropout"])
    raise ValueError(f"Unknown model: {name!r}. Choose 'rnn', 'lstm', or 'tcn'.")

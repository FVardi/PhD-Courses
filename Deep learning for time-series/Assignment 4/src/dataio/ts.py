"""Parser for the UCR/UEA .ts format.

.ts is the only format the archive ships for every dataset in this assignment: the
univariate problems also carry a _TRAIN.txt, but ArticularyWordRecognition and Epilepsy
do not, so anything built on .txt would work for the Ford datasets only.
"""

from pathlib import Path

import numpy as np


def read_header(path: Path) -> dict:
    """Read the @-block without touching the data rows."""
    header = {
        "problem_name": None,
        "univariate": True,
        "channels": 1,
        "length": None,
        "equal_length": None,
        "missing": None,
        "class_order": [],
    }
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            low = line.lower()
            if low.startswith("@data"):
                break
            parts = line.split()
            if low.startswith("@problemname"):
                header["problem_name"] = parts[1]
            elif low.startswith("@univariate"):
                header["univariate"] = parts[1].lower() == "true"
            elif low.startswith("@dimensions"):
                header["channels"] = int(parts[1])
            elif low.startswith("@serieslength"):
                header["length"] = int(parts[1])
            elif low.startswith("@equallength"):
                header["equal_length"] = parts[1].lower() == "true"
            elif low.startswith("@missing"):
                header["missing"] = parts[1].lower() == "true"
            elif low.startswith("@classlabel") and len(parts) > 2:
                header["class_order"] = parts[2:]
    return header


def read_ts(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return X of shape (n_series, n_channels, length), raw string labels, and the header.

    Channels within a row are separated by ':' and the label is the final field. '?' marks
    a missing value and becomes NaN.
    """
    path = Path(path)
    header = read_header(path)
    series, labels = [], []
    in_data = False

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            if not line or line.startswith("#"):
                continue
            *channels, label = line.split(":")
            series.append(
                [
                    [np.nan if v.strip() == "?" else float(v) for v in channel.split(",")]
                    for channel in channels
                ]
            )
            labels.append(label.strip())

    lengths = {len(c) for row in series for c in row}
    if len(lengths) != 1:
        raise ValueError(f"{path.name}: unequal series lengths {sorted(lengths)}")

    return np.asarray(series, dtype=np.float64), np.asarray(labels), header


def encode_labels(labels: np.ndarray, class_order: list[str]) -> np.ndarray:
    """Map string labels to integers using the header's declared class order.

    Deriving the mapping from the header rather than from the values present means train
    and test always agree, even if a split were missing a class.
    """
    lookup = {name: i for i, name in enumerate(class_order)}
    unknown = set(labels) - lookup.keys()
    if unknown:
        raise ValueError(f"labels absent from the header's @classLabel: {sorted(unknown)}")
    return np.asarray([lookup[v] for v in labels], dtype=np.int64)

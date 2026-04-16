"""
Quick inspection of raw dataset column names and sample values.
"""

import os
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
DATA_RAW     = os.path.join(PROJECT_DIR, "data", "london_smart_meters")
BLOCKS_DIR   = os.path.join(DATA_RAW, "halfhourly_dataset", "halfhourly_dataset")

DATASETS = {
    "halfhourly":            os.path.join(BLOCKS_DIR, "block_0.csv"),
    "daily":                 os.path.join(DATA_RAW, "daily_dataset",   "daily_dataset",   "block_0.csv"),
    "hhblock":               os.path.join(DATA_RAW, "hhblock_dataset", "hhblock_dataset", "block_0.csv"),
    "acorn":                 os.path.join(DATA_RAW, "acorn_details.csv"),
    "informations_households": os.path.join(DATA_RAW, "informations_households.csv"),
}


def inspect_datasets(n: int = 3) -> None:
    """Print column names and first n rows for each raw dataset."""
    for name, path in DATASETS.items():
        print(f"\n{'='*60}\n  {name}  —  {os.path.basename(path)}\n{'='*60}")
        df = pd.read_csv(path, nrows=n, encoding="latin-1")
        print("Columns:", df.columns.tolist())
        print(df.to_string(index=False))


if __name__ == "__main__":
    inspect_datasets()

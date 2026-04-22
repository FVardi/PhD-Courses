"""
Task 03: Feature Engineering — Lag and Rolling Features

Reads the imputed halfhourly dataset in household batches to stay within memory,
computes leakage-safe features within each household, and writes the result as a
partitioned parquet directory at data/features/.

Feature sets (all strictly causal — past-only, within-household):
  Lag features    : lags 1–6, 48, 336 slots
  Rolling stats   : mean and std over windows of 6, 48, 336 slots

To add more features: implement a builder in src/feature_engineering/builders.py
and append a partial of it to FEATURE_BUILDERS below.

Reusable logic lives in:
  src/feature_engineering/builders.py   (build_lag_features, build_rolling_features)
"""

# %%
import logging
import sys
from functools import partial
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.builders import (
    build_lag_features,
    build_rolling_features,
    build_ewma_features,
    build_calendar_features,
    build_fourier_features,
)

# %%

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMPUTED_PATH = PROJECT_ROOT / "data" / "preprocessed" / "halfhourly_imputed.parquet"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
VALUE_COL    = "energy_imputed_seasonal"

# ── Run mode ──────────────────────────────────────────────────────────────────
# QUICK_RUN=True  → process only the first 50 households; fast for inspection.
# QUICK_RUN=False → process all households; required before tasks 04–10.
QUICK_RUN    = False
MAX_HH_QUICK = 50     # households to process in quick mode
BATCH_SIZE   = 50     # households per batch; tune down if memory is tight

# %%
# --- Feature configuration ----------------------------------------------------
# Add new builders here as more features are developed.

FEATURE_BUILDERS = [
    partial(build_lag_features,     lags=[1, 2, 3, 4, 5, 6, 48, 336]),
    partial(build_rolling_features, windows=[6, 48, 336]),
    partial(build_ewma_features,    spans=[6, 48, 336]),
    build_calendar_features,
    partial(build_fourier_features, periods=[48, 336], n_terms=2),
]

# %%
# --- Step 1: Discover all households without loading the full file -------------

logger.info("Reading household index from %s …", IMPUTED_PATH)
all_lclids = (
    pd.read_parquet(IMPUTED_PATH, columns=[])
    .index.get_level_values("LCLid")
    .unique()
    .tolist()
)
if QUICK_RUN:
    all_lclids = all_lclids[:MAX_HH_QUICK]
    logger.info("QUICK_RUN: limiting to %d households", len(all_lclids))

n_households = len(all_lclids)
n_batches    = (n_households + BATCH_SIZE - 1) // BATCH_SIZE
logger.info(
    "Found %d households → %d batches of up to %d",
    n_households, n_batches, BATCH_SIZE,
)

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# %%
# --- Step 2: Process batches --------------------------------------------------
#
# Each batch is read from parquet with row pushdown, all feature builders are
# applied in sequence, and the result is written as a numbered part file.
# The output directory can be read back as one dataset:
#   pd.read_parquet("data/features/")

for batch_idx in range(n_batches):
    batch_ids = all_lclids[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

    batch = pd.read_parquet(
        IMPUTED_PATH,
        filters=[("LCLid", "in", batch_ids)],
    )

    feature_frames = [builder(batch, VALUE_COL) for builder in FEATURE_BUILDERS]
    batch = pd.concat([batch, *feature_frames], axis=1)

    out_path = FEATURES_DIR / f"part_{batch_idx:04d}.parquet"
    batch.to_parquet(out_path)
    logger.info(
        "Batch %d / %d  |  %d households  |  %d rows  |  %d feature cols  →  %s",
        batch_idx + 1, n_batches, len(batch_ids), len(batch),
        len(batch.columns), out_path.name,
    )

logger.info("Done. %d part files written to %s", n_batches, FEATURES_DIR)

# %%


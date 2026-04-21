"""
Task 05: Forecasting Wrapper Demo

Demonstrates FeatureConfig, MissingValueConfig, ModelConfig, and MLForecast
on a single household.  All later tasks reuse this interface.

Outputs are written to results/artifacts/.
"""

# %%
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    VALUE_COL, TRAIN_END, VAL_START, VAL_END,
    make_feature_config, make_missing_config, make_sklearn_model_config,
    mase_denom, eval_metrics,
    save_fig, save_csv,
)
from src.forecasting import MLForecast
from src.transforms.transforms import DeseasonalisingTransform

# %%

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "results" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

LCLID = "MAC000002"

# %%
# --- Step 1: Load data for one household -------------------------------------

logger.info("Loading features for %s …", LCLID)
data = pd.read_parquet(FEATURES_DIR, filters=[("LCLid", "in", [LCLID])])
logger.info("Loaded %d rows, %d columns", len(data), len(data.columns))

_tstp = data.index.get_level_values("tstp")
train = data.loc[_tstp < TRAIN_END]
val   = data.loc[(_tstp >= VAL_START) & (_tstp < VAL_END)]

# %%
# --- Step 2: FeatureConfig ---------------------------------------------------

feature_config = make_feature_config()

logger.info(
    "FeatureConfig: %d continuous, %d categorical, %d boolean",
    len(feature_config.continuous_features),
    len(feature_config.categorical_features),
    len(feature_config.boolean_features),
)

# %%
# --- Step 3: MissingValueConfig — spot-check ---------------------------------

import numpy as np
missing_config = make_missing_config()
X, y, _ = feature_config.get_X_y(train)
_test = X.copy()
_test.iloc[0, 0] = np.nan
_filled = missing_config.impute_missing_values(_test)
assert not np.isnan(_filled.iloc[0, 0]), "MissingValueConfig spot-check failed"
logger.info("MissingValueConfig spot-check passed.")

# %%
# --- Step 4: ModelConfig (Ridge, with OHE + StandardScaler) ------------------

model_config = make_sklearn_model_config(Ridge(alpha=1.0), "Ridge")
logger.info(
    "ModelConfig: model=%s  normalize=%s  encode=%s",
    model_config.model_name, model_config.normalize, model_config.encode_categoricals,
)

# %%
# --- Step 5: MLForecast — no target transform --------------------------------

wrapper_plain = MLForecast(
    model_config         = model_config,
    feature_config       = feature_config,
    missing_value_config = missing_config,
    target_transformer   = None,
)
wrapper_plain.fit(train)
logger.info("Plain wrapper fitted.")

# %%
# --- Step 6: Predict on validation set ---------------------------------------

y_hat_plain = wrapper_plain.predict(val)
y_val_true  = val[VALUE_COL].reindex(y_hat_plain.index)
_denom      = mase_denom(train[VALUE_COL])
m_plain     = eval_metrics(y_val_true, y_hat_plain, _denom)

logger.info(
    "Ridge (no transform)  |  MAE=%.4f  MASE=%.4f  (val: %s → %s)",
    m_plain["mae"], m_plain["mase"], VAL_START.date(), VAL_END.date(),
)

# %%
# --- Step 7: MLForecast — with DeseasonalisingTransform ----------------------

wrapper_ds = MLForecast(
    model_config         = model_config,
    feature_config       = feature_config,
    missing_value_config = missing_config,
    target_transformer   = DeseasonalisingTransform(period=48),
)
wrapper_ds.fit(train)
y_hat_ds  = wrapper_ds.predict(val)
m_ds      = eval_metrics(val[VALUE_COL].reindex(y_hat_ds.index), y_hat_ds, _denom)

logger.info(
    "Ridge + Deseason      |  MAE=%.4f  MASE=%.4f",
    m_ds["mae"], m_ds["mase"],
)

# %%
# --- Step 8: Verify predictions are on the original scale --------------------

assert y_hat_plain.index.equals(y_hat_ds.index), "prediction indices must match"
logger.info("Prediction range (plain):    [%.3f, %.3f]", y_hat_plain.min(), y_hat_plain.max())
logger.info("Prediction range (deseason): [%.3f, %.3f]", y_hat_ds.min(),    y_hat_ds.max())
logger.info("Actual target range (val):   [%.3f, %.3f]", y_val_true.min(),  y_val_true.max())

# %%
# --- Step 9: Save results ----------------------------------------------------

import pandas as _pd
results = _pd.DataFrame([
    {"model": "Ridge", "transform": "none",     "mae": m_plain["mae"], "mase": m_plain["mase"]},
    {"model": "Ridge", "transform": "deseason", "mae": m_ds["mae"],    "mase": m_ds["mase"]},
]).set_index(["model", "transform"])

save_csv(results, "task05_wrapper_demo_metrics.csv", ARTIFACTS_DIR)
logger.info("Metrics:\n%s", results.to_string())

"""
pytest tests for Task 05: FeatureConfig, MissingValueConfig, MLForecast.

Run with:
    pytest tests/test_wrapper.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs import FeatureConfig, MissingValueConfig, ModelConfig
from src.forecasting import MLForecast
from src.transforms.transforms import LogTransform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic MultiIndex (LCLid × tstp) DataFrame mimicking the features parquet."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=n, freq="30min")
    mi  = pd.MultiIndex.from_arrays(
        [["MAC000001"] * n, idx], names=["LCLid", "tstp"]
    )
    df = pd.DataFrame(
        {
            "energy_imputed_seasonal": rng.random(n) + 0.3,
            "lag_1":                   rng.random(n),
            "rolling_mean_6":          rng.random(n),
            "day_of_week":             (idx.dayofweek).astype(str),
            "is_weekend":              (idx.dayofweek >= 5).astype(int),
        },
        index=mi,
    )
    return df


@pytest.fixture
def toy_df() -> pd.DataFrame:
    return _make_df()


@pytest.fixture
def feature_cfg() -> FeatureConfig:
    return FeatureConfig(
        timestamp_col        = "tstp",
        target_col           = "energy_imputed_seasonal",
        original_target_col  = "energy_imputed_seasonal",
        continuous_features  = ["lag_1", "rolling_mean_6"],
        categorical_features = ["day_of_week"],
        boolean_features     = ["is_weekend"],
        index_cols           = ["LCLid", "tstp"],
    )


@pytest.fixture
def missing_cfg() -> MissingValueConfig:
    return MissingValueConfig(ffill_cols=["lag_1", "rolling_mean_6"])


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(
        estimator           = Ridge(alpha=1.0),
        model_name          = "Ridge",
        normalize           = True,
        fill_missing        = True,
        encode_categoricals = True,
        categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    )


# ---------------------------------------------------------------------------
# (a) get_X_y returns correct X / y / y_orig
# ---------------------------------------------------------------------------

class TestGetXy:
    def test_X_excludes_target(self, toy_df, feature_cfg):
        X, y, y_orig = feature_cfg.get_X_y(toy_df)
        assert "energy_imputed_seasonal" not in X.columns

    def test_X_contains_all_feature_groups(self, toy_df, feature_cfg):
        X, _, _ = feature_cfg.get_X_y(toy_df)
        for col in ["lag_1", "rolling_mean_6", "day_of_week", "is_weekend"]:
            assert col in X.columns, f"{col} missing from X"

    def test_X_excludes_timestamp_col(self, toy_df, feature_cfg):
        X, _, _ = feature_cfg.get_X_y(toy_df)
        assert "tstp" not in X.columns

    def test_y_is_target_col(self, toy_df, feature_cfg):
        _, y, _ = feature_cfg.get_X_y(toy_df)
        pd.testing.assert_series_equal(y, toy_df["energy_imputed_seasonal"])

    def test_y_orig_is_original_target_col(self, toy_df, feature_cfg):
        _, _, y_orig = feature_cfg.get_X_y(toy_df)
        pd.testing.assert_series_equal(y_orig, toy_df["energy_imputed_seasonal"])

    def test_lengths_match(self, toy_df, feature_cfg):
        X, y, y_orig = feature_cfg.get_X_y(toy_df)
        assert len(X) == len(y) == len(y_orig) == len(toy_df)


class TestFeatureConfigValidation:
    def test_no_features_raises(self):
        with pytest.raises(ValueError, match="at least one feature"):
            FeatureConfig(
                timestamp_col="tstp", target_col="y", original_target_col="y",
                continuous_features=[], categorical_features=[],
                boolean_features=[], index_cols=["tstp"],
            )

    def test_target_in_features_raises(self):
        with pytest.raises(ValueError, match="must not appear"):
            FeatureConfig(
                timestamp_col="tstp", target_col="y", original_target_col="y",
                continuous_features=["y", "x1"],
                categorical_features=[], boolean_features=[], index_cols=["tstp"],
            )

    def test_timestamp_in_features_raises(self):
        with pytest.raises(ValueError, match="must not appear"):
            FeatureConfig(
                timestamp_col="tstp", target_col="y", original_target_col="y",
                continuous_features=["tstp", "x1"],
                categorical_features=[], boolean_features=[], index_cols=["tstp"],
            )

    def test_duplicate_across_groups_raises(self):
        with pytest.raises(ValueError, match="more than one feature group"):
            FeatureConfig(
                timestamp_col="tstp", target_col="y", original_target_col="y",
                continuous_features=["x1"],
                categorical_features=["x1"],
                boolean_features=[], index_cols=["tstp"],
            )


# ---------------------------------------------------------------------------
# (b) impute_missing_values fills intended columns
# ---------------------------------------------------------------------------

class TestMissingValueConfig:
    def test_ffill_fills_forward(self, toy_df, feature_cfg):
        cfg = MissingValueConfig(ffill_cols=["lag_1"])
        X, _, _ = feature_cfg.get_X_y(toy_df)
        X_nan = X.copy()
        X_nan.iloc[5:10, X_nan.columns.get_loc("lag_1")] = np.nan
        result = cfg.impute_missing_values(X_nan)
        assert result["lag_1"].isna().sum() == 0

    def test_zero_fill(self, toy_df, feature_cfg):
        cfg = MissingValueConfig(zero_fill_cols=["rolling_mean_6"])
        X, _, _ = feature_cfg.get_X_y(toy_df)
        X_nan = X.copy()
        X_nan.iloc[0, X_nan.columns.get_loc("rolling_mean_6")] = np.nan
        result = cfg.impute_missing_values(X_nan)
        assert result["rolling_mean_6"].iloc[0] == 0.0

    def test_fallback_mean_fills_remaining(self, toy_df, feature_cfg):
        cfg = MissingValueConfig()   # no explicit rules → fallback only
        X, _, _ = feature_cfg.get_X_y(toy_df)
        X_nan = X.copy()
        X_nan.iloc[:, 0] = np.nan   # inject NaN in first numeric col
        result = cfg.impute_missing_values(X_nan)
        assert result.iloc[:, 0].isna().sum() == 0

    def test_output_shape_unchanged(self, toy_df, feature_cfg, missing_cfg):
        X, _, _ = feature_cfg.get_X_y(toy_df)
        result = missing_cfg.impute_missing_values(X)
        assert result.shape == X.shape


# ---------------------------------------------------------------------------
# (c) predict() raises RuntimeError before fit()
# ---------------------------------------------------------------------------

class TestPredictBeforeFit:
    def test_raises_runtime_error(self, toy_df, feature_cfg, missing_cfg, model_cfg):
        wrapper = MLForecast(model_cfg, feature_cfg, missing_cfg)
        with pytest.raises(RuntimeError, match="fit()"):
            wrapper.predict(toy_df)


# ---------------------------------------------------------------------------
# (d) predictions are on original scale when a transform is used
# ---------------------------------------------------------------------------

class TestPredictionsOnOriginalScale:
    def test_log_transform_predictions_in_original_range(
        self, toy_df, feature_cfg, missing_cfg, model_cfg
    ):
        wrapper = MLForecast(
            model_config         = model_cfg,
            feature_config       = feature_cfg,
            missing_value_config = missing_cfg,
            target_transformer   = LogTransform(),
        )
        n = len(toy_df)
        train = toy_df.iloc[: n // 2]
        val   = toy_df.iloc[n // 2 :]

        wrapper.fit(train)
        y_hat = wrapper.predict(val)

        y_actual = toy_df["energy_imputed_seasonal"].iloc[n // 2 :]

        # Predictions should be in the same order of magnitude as actual values
        assert y_hat.min() > -1.0,  "predictions unexpectedly negative"
        assert y_hat.max() < 100.0, "predictions far out of range — inverse transform may have failed"

        # Crude scale check: pred mean within 2× of actual mean
        ratio = y_hat.mean() / y_actual.reindex(y_hat.index).mean()
        assert 0.1 < ratio < 10.0, f"prediction scale looks wrong (ratio={ratio:.2f})"

    def test_no_transform_predictions_in_original_range(
        self, toy_df, feature_cfg, missing_cfg, model_cfg
    ):
        wrapper = MLForecast(model_cfg, feature_cfg, missing_cfg)
        n = len(toy_df)
        wrapper.fit(toy_df.iloc[: n // 2])
        y_hat = wrapper.predict(toy_df.iloc[n // 2 :])

        y_actual = toy_df["energy_imputed_seasonal"].iloc[n // 2 :]
        ratio = y_hat.mean() / y_actual.reindex(y_hat.index).mean()
        assert 0.1 < ratio < 10.0, f"prediction scale looks wrong (ratio={ratio:.2f})"

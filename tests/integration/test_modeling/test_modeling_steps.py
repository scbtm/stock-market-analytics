"""
Integration tests for modeling workflow steps.

Tests the complete modeling workflow step functions, focusing on
end-to-end functionality while mocking external dependencies.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import pandas as pd
import numpy as np

from stock_market_analytics.modeling import modeling_steps


class TestModelingStepsIntegration:
    """Integration tests for modeling workflow steps."""

    @pytest.fixture
    def mock_environment(self):
        """Set up mock environment for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"BASE_DATA_PATH": temp_dir}):
                yield temp_dir

    @pytest.fixture
    def sample_features_data(self):
        """Sample features data for model training."""
        np.random.seed(42)
        n_samples = 1000

        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        # Create data with all required columns from config
        from stock_market_analytics.config import config

        data = {}
        # Add required metadata columns
        data["date"] = dates
        data["symbol"] = ["AAPL"] * n_samples

        # Add all the feature columns from config
        for feature in config.modeling.features:
            if feature not in ["date", "symbol"]:  # Skip already added columns
                data[feature] = np.random.randn(n_samples)

        # Add the target column
        data[config.modeling.target] = np.random.normal(0, 0.02, n_samples)

        df = pd.DataFrame(data)
        # Don't set date as index - keep it as a column for modeling
        return df

    def test_load_features_data_success(self, mock_environment, sample_features_data):
        """Test successful features data loading."""
        # Create features file with correct name
        features_path = Path(mock_environment) / "stock_history_features.parquet"
        sample_features_data.to_parquet(features_path)

        # Test loading
        result = modeling_steps.load_features_data(Path(mock_environment))

        assert len(result) == 1000
        assert "y_log_returns" in result.columns
        assert "symbol" in result.columns

    def test_load_features_data_missing_file(self, mock_environment):
        """Test loading features when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            modeling_steps.load_features_data(Path(mock_environment))

    def test_prepare_modeling_data(self, sample_features_data):
        """Test data preparation for modeling."""
        result = modeling_steps.prepare_modeling_data(sample_features_data)

        # Should have train, val, cal, test splits
        assert "train" in result
        assert "val" in result
        assert "cal" in result
        assert "test" in result

        # Each split should have X and y
        for split_name, (X, y) in result.items():
            assert len(X) == len(y)
            assert not X.empty
            assert not y.empty

    def test_get_adhoc_transforms_pipeline(self):
        """Test getting transformation pipeline."""
        pipeline = modeling_steps.get_adhoc_transforms_pipeline()

        # Should return a sklearn transformer
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "transform")

    def test_get_catboost_multiquantile_model(self):
        """Test getting CatBoost model."""
        model = modeling_steps.get_catboost_multiquantile_model()

        # Should return a CatBoost model
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_calibrator(self):
        """Test getting calibrator."""
        calibrator = modeling_steps.get_calibrator()

        # Should return a calibrator with fit and transform methods
        assert hasattr(calibrator, "fit")
        assert hasattr(calibrator, "transform")

    def test_get_evaluator(self):
        """Test getting evaluator."""
        evaluator = modeling_steps.get_evaluator()

        # Should return an evaluator with evaluation methods
        assert hasattr(evaluator, "evaluate_intervals")
        assert hasattr(evaluator, "evaluate_quantiles")

"""
Integration tests for monitoring workflow steps.

Tests the complete monitoring workflow step functions, focusing on
end-to-end functionality while mocking external dependencies.
"""

import os
import tempfile
from unittest.mock import Mock, patch
import pytest
import pandas as pd
import numpy as np

from stock_market_analytics.monitoring import monitoring_steps


class TestMonitoringStepsIntegration:
    """Integration tests for monitoring workflow steps."""

    @pytest.fixture
    def mock_environment(self):
        """Set up mock environment for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"BASE_DATA_PATH": temp_dir}):
                yield temp_dir

    @pytest.fixture
    def sample_monitoring_data(self):
        """Sample data for monitoring."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        feature_cols = [f"feature_{i}" for i in range(n_features)]
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="D")

        feature_data = np.random.randn(n_samples, n_features)
        target = np.random.normal(0, 0.02, n_samples)

        df = pd.DataFrame(feature_data, columns=feature_cols, index=dates)
        df["log_returns_1d"] = target
        df["date"] = dates

        return df

    @patch("wandb.Api")
    def test_download_artifacts_success(self, mock_wandb_api):
        """Test successful artifact download."""
        # Mock W&B API
        mock_api = Mock()
        mock_run = Mock()
        mock_artifact = Mock()
        mock_artifact.download.return_value = "/tmp/artifact_dir"
        mock_run.logged_artifacts.return_value = [mock_artifact]
        mock_api.runs.return_value = [mock_run]
        mock_wandb_api.return_value = mock_api

        # Test download
        result = monitoring_steps.download_artifacts()

        assert len(result) == 4  # model_dir, model_name, dataset_dir, dataset_name

    def test_get_covariate_drift_metrics(self, sample_monitoring_data):
        """Test covariate drift calculation."""
        reference_data = sample_monitoring_data.iloc[:50]
        current_data = sample_monitoring_data.iloc[50:]
        feature_columns = [f"feature_{i}" for i in range(10)]

        result = monitoring_steps.get_covariate_drift_metrics(
            reference_df=reference_data,
            current_df=current_data,
            feature_columns=feature_columns,
        )

        assert "aggregate" in result
        assert "per_feature" in result
        assert "mean_psi" in result["aggregate"]

    def test_get_target_drift_metrics(self, sample_monitoring_data):
        """Test target drift calculation."""
        reference_targets = sample_monitoring_data["log_returns_1d"].iloc[:50]
        current_targets = sample_monitoring_data["log_returns_1d"].iloc[50:]

        result = monitoring_steps.get_target_drift_metrics(
            reference_targets=reference_targets, current_targets=current_targets
        )

        assert "distribution_tests" in result
        assert "psi" in result["distribution_tests"]

    def test_get_predicted_quantiles_metrics(self):
        """Test quantile prediction metrics."""
        np.random.seed(42)
        n_samples = 100
        n_quantiles = 5

        y_true = np.random.normal(0, 0.02, n_samples)
        y_pred_quantiles = pd.DataFrame(
            np.random.randn(n_samples, n_quantiles),
            columns=[f"q_{q:.2f}" for q in [0.1, 0.25, 0.5, 0.75, 0.9]],
        )
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        result = monitoring_steps.get_predicted_quantiles_metrics(
            y_true=pd.Series(y_true),
            y_pred_quantiles=y_pred_quantiles,
            quantiles=quantiles,
        )

        assert "pinball_losses" in result
        assert "distributional" in result
        assert "coverage" in result

    def test_get_calibration_metrics(self):
        """Test calibration metrics calculation."""
        np.random.seed(42)
        n_samples = 100

        y_true = np.random.normal(0, 0.02, n_samples)
        y_lower = y_true - 0.05
        y_upper = y_true + 0.05

        result = monitoring_steps.get_calibration_metrics(
            y_true=pd.Series(y_true),
            y_lower=pd.Series(y_lower),
            y_upper=pd.Series(y_upper),
            confidence_level=0.8,
        )

        assert "coverage" in result
        assert "interval_width" in result
        assert "scoring" in result

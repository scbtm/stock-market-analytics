"""
Integration tests for inference steps module.

These tests verify the integration between inference components and external systems
like Weights & Biases, model loading workflows, and end-to-end inference pipelines.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import os

from stock_market_analytics.inference.inference_steps import (
    download_artifacts,
    load_model,
    download_and_load_model,
    get_inference_data,
    make_prediction_intervals,
    predict_quantiles,
)


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def mock_wandb_api():
    """Mock Weights & Biases API for integration testing."""
    with patch('stock_market_analytics.inference.inference_steps.wandb.Api') as mock_api:
        api_instance = Mock()
        mock_api.return_value = api_instance

        # Mock artifact with realistic behavior
        artifact = Mock()
        artifact.download.return_value = "/tmp/model_dir"
        artifact.name = "pipeline:latest"
        artifact.version = "latest"
        api_instance.artifact.return_value = artifact

        yield api_instance


@pytest.fixture
def mock_joblib():
    """Mock joblib for integration testing."""
    with patch('stock_market_analytics.inference.inference_steps.joblib.load') as mock_load:
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.1, 0.9], [0.2, 0.8]])

        # Mock pipeline structure for predict_quantiles
        transforms = Mock()
        transforms.transform.return_value = np.array([[1, 2], [3, 4]])

        regressor = Mock()
        regressor.predict.return_value = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]])

        mock_model.named_steps = {
            'transformations': transforms,
            'model': regressor
        }

        mock_load.return_value = mock_model
        yield mock_load


@pytest.fixture
def mock_config():
    """Mock configuration for integration testing."""
    with patch('stock_market_analytics.inference.inference_steps.config') as mock_cfg:
        mock_cfg.modeling.features = ['feature1', 'feature2']
        mock_cfg.modeling.quantiles = [0.1, 0.5, 0.9]
        yield mock_cfg


# -------------------------
# Integration Tests
# -------------------------

class TestWandBArtifactIntegration:
    """Integration tests for W&B artifact downloading workflow."""

    def test_download_and_load_model_full_workflow(self, mock_wandb_api, mock_joblib):
        """Test complete download and load workflow integration."""
        with patch.dict(os.environ, {'WANDB_API_KEY': 'test_key', 'MODEL_NAME': 'pipeline:v1.0'}):
            # Test the full workflow
            result = download_and_load_model()

            # Verify W&B API interaction
            mock_wandb_api.artifact.assert_called_once_with(
                "san-cbtm/stock-market-analytics/pipeline:v1.0", type="model"
            )

            # Verify artifact download
            artifact = mock_wandb_api.artifact.return_value
            artifact.download.assert_called_once()

            # Verify model loading
            mock_joblib.assert_called_once_with("/tmp/model_dir/pipeline.pkl")

            # Verify result
            assert result == mock_joblib.return_value

    def test_download_artifacts_environment_integration(self, mock_wandb_api):
        """Test artifact download with different environment configurations."""
        test_cases = [
            ({'WANDB_API_KEY': 'key1'}, "pipeline:latest"),
            ({'WANDB_API_KEY': 'key2', 'MODEL_NAME': 'custom:v2.0'}, "custom:v2.0"),
            ({'WANDB_API_KEY': 'key3', 'MODEL_NAME': 'pipeline'}, "pipeline"),
        ]

        for env_vars, expected_model in test_cases:
            with patch.dict(os.environ, env_vars, clear=True):
                model_dir, model_name = download_artifacts()

                assert model_dir == "/tmp/model_dir"
                assert model_name == expected_model

                # Verify correct artifact path construction
                expected_artifact_path = f"san-cbtm/stock-market-analytics/{expected_model}"
                mock_wandb_api.artifact.assert_called_with(expected_artifact_path, type="model")

    def test_model_loading_integration_different_formats(self, mock_joblib):
        """Test model loading integration with different model name formats."""
        test_cases = [
            ("pipeline:latest", "/tmp/model/pipeline.pkl"),
            ("custom_model:v1.0", "/tmp/model/custom_model.pkl"),
            ("simple_name", "/tmp/model/simple_name.pkl"),
        ]

        for model_name, expected_path in test_cases:
            result = load_model("/tmp/model", model_name)

            mock_joblib.assert_called_with(expected_path)
            assert result == mock_joblib.return_value


class TestInferenceDataIntegration:
    """Integration tests for inference data collection and feature generation."""

    @patch('stock_market_analytics.inference.inference_steps.generate_inference_features')
    @patch('stock_market_analytics.inference.inference_steps.collect_inference_data')
    def test_get_inference_data_full_pipeline(self, mock_collect, mock_generate):
        """Test complete inference data pipeline integration."""
        # Mock realistic data flow
        mock_raw_data = Mock()
        mock_raw_data.shape = (10, 5)
        mock_collect.return_value = mock_raw_data

        mock_features = Mock()
        mock_features.shape = (10, 8)
        mock_features_df = pd.DataFrame({
            'feature1': range(10),
            'feature2': range(10, 20),
            'date': pd.date_range('2023-01-01', periods=10)
        })
        mock_features.to_pandas.return_value = mock_features_df
        mock_generate.return_value = mock_features

        # Test with different symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]

        for symbol in symbols:
            result = get_inference_data(symbol)

            # Verify data collection step
            mock_collect.assert_called_with(symbol)

            # Verify feature generation step
            mock_generate.assert_called_with(mock_raw_data)

            # Verify data conversion
            mock_features.to_pandas.assert_called()

            # Verify result format
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 10
            assert 'feature1' in result.columns

    @patch('stock_market_analytics.inference.inference_steps.generate_inference_features')
    @patch('stock_market_analytics.inference.inference_steps.collect_inference_data')
    def test_get_inference_data_error_handling_integration(self, mock_collect, mock_generate):
        """Test error handling in inference data pipeline integration."""
        # Test data collection failure
        mock_collect.side_effect = Exception("Data collection service unavailable")

        with pytest.raises(Exception, match="Data collection service unavailable"):
            get_inference_data("AAPL")

        # Reset and test feature generation failure
        mock_collect.side_effect = None
        mock_collect.return_value = Mock()
        mock_generate.side_effect = Exception("Feature generation pipeline failed")

        with pytest.raises(Exception, match="Feature generation pipeline failed"):
            get_inference_data("AAPL")

    @patch('stock_market_analytics.inference.inference_steps.generate_inference_features')
    @patch('stock_market_analytics.inference.inference_steps.collect_inference_data')
    def test_get_inference_data_empty_data_integration(self, mock_collect, mock_generate):
        """Test inference pipeline with empty data integration."""
        # Mock empty data scenario
        mock_collect.return_value = Mock()

        mock_features = Mock()
        mock_features.shape = (0, 8)
        empty_df = pd.DataFrame(columns=['feature1', 'feature2'])
        mock_features.to_pandas.return_value = empty_df
        mock_generate.return_value = mock_features

        result = get_inference_data("UNKNOWN_SYMBOL")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ['feature1', 'feature2']


class TestPredictionIntegration:
    """Integration tests for prediction workflows."""

    def test_make_prediction_intervals_workflow_integration(self, mock_config):
        """Test prediction intervals workflow integration."""
        # Create realistic mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([
            [0.1, 0.9],  # Sample 1: low_quantile, high_quantile
            [0.2, 0.8],  # Sample 2
            [0.15, 0.85] # Sample 3
        ])

        # Create realistic input data
        input_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'extra_column': ['A', 'B', 'C']  # Should be ignored
        })

        result = make_prediction_intervals(mock_model, input_data.copy())

        # Verify model was called with correct features only
        model_input = mock_model.predict.call_args[0][0]
        expected_features = input_data[mock_config.modeling.features]
        pd.testing.assert_frame_equal(model_input, expected_features)

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'pred_low_quantile' in result.columns
        assert 'pred_high_quantile' in result.columns
        assert 'date' in result.columns  # Original columns preserved
        assert 'extra_column' in result.columns

        # Verify predictions
        np.testing.assert_array_equal(result['pred_low_quantile'].values, [0.1, 0.2, 0.15])
        np.testing.assert_array_equal(result['pred_high_quantile'].values, [0.9, 0.8, 0.85])

    def test_predict_quantiles_pipeline_integration(self, mock_config):
        """Test quantile prediction pipeline integration."""
        # Create realistic mock model with pipeline structure
        mock_model = Mock()

        # Mock transformation step
        transforms = Mock()
        transforms.transform.return_value = np.array([
            [1.1, 4.1],
            [2.1, 5.1],
            [3.1, 6.1]
        ])

        # Mock regressor step
        regressor = Mock()
        regressor.predict.return_value = np.array([
            [0.1, 0.5, 0.9],  # Sample 1: q10, q50, q90
            [0.2, 0.4, 0.8],  # Sample 2
            [0.15, 0.6, 0.85] # Sample 3
        ])

        mock_model.named_steps = {
            'transformations': transforms,
            'model': regressor
        }

        # Create input data
        input_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })

        result = predict_quantiles(mock_model, input_data.copy())

        # Verify pipeline integration
        transforms.transform.assert_called_once()
        regressor.predict.assert_called_once_with(
            transforms.transform.return_value,
            return_full_quantiles=True
        )

        # Verify feature selection for transformation
        transform_input = transforms.transform.call_args[0][0]
        expected_features = input_data[mock_config.modeling.features]
        pd.testing.assert_frame_equal(transform_input, expected_features)

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

        # Verify quantile columns
        expected_quantile_cols = ['pred_Q_10', 'pred_Q_50', 'pred_Q_90']
        for col in expected_quantile_cols:
            assert col in result.columns

        # Verify predictions
        np.testing.assert_array_equal(result['pred_Q_10'].values, [0.1, 0.2, 0.15])
        np.testing.assert_array_equal(result['pred_Q_50'].values, [0.5, 0.4, 0.6])
        np.testing.assert_array_equal(result['pred_Q_90'].values, [0.9, 0.8, 0.85])

        # Verify original columns preserved
        assert 'symbol' in result.columns
        assert 'date' in result.columns

    def test_prediction_workflow_with_empty_data_integration(self, mock_config):
        """Test prediction workflows with empty data integration."""
        mock_model = Mock()

        # Test make_prediction_intervals with empty data
        mock_model.predict.return_value = np.array([]).reshape(0, 2)
        empty_df = pd.DataFrame(columns=['feature1', 'feature2'])

        result = make_prediction_intervals(mock_model, empty_df)
        assert len(result) == 0
        assert 'pred_low_quantile' in result.columns
        assert 'pred_high_quantile' in result.columns

        # Test predict_quantiles with empty data
        transforms = Mock()
        transforms.transform.return_value = np.array([]).reshape(0, 2)
        regressor = Mock()
        regressor.predict.return_value = np.array([]).reshape(0, 3)

        mock_model.named_steps = {
            'transformations': transforms,
            'model': regressor
        }

        result = predict_quantiles(mock_model, empty_df)
        assert len(result) == 0
        expected_cols = ['pred_Q_10', 'pred_Q_50', 'pred_Q_90']
        for col in expected_cols:
            assert col in result.columns
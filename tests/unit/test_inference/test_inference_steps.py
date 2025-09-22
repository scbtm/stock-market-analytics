"""
Unit tests for inference_steps module.

This module tests the core inference pipeline functions including
artifact downloading, model loading, and prediction generation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
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
    """Mock Weights & Biases API."""
    with patch(
        "stock_market_analytics.inference.inference_steps.wandb.Api"
    ) as mock_api:
        api_instance = Mock()
        mock_api.return_value = api_instance

        # Mock artifact
        artifact = Mock()
        artifact.download.return_value = "/tmp/model_dir"
        api_instance.artifact.return_value = artifact

        yield api_instance


@pytest.fixture
def mock_joblib():
    """Mock joblib for model loading."""
    with patch(
        "stock_market_analytics.inference.inference_steps.joblib.load"
    ) as mock_load:
        mock_model = Mock()
        mock_load.return_value = mock_model
        yield mock_load


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        }
    )


@pytest.fixture
def mock_model():
    """Mock model with predict method."""
    model = Mock()
    model.predict.return_value = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])

    # Mock named_steps for pipeline
    transforms = Mock()
    transforms.transform.return_value = np.array([[1, 2], [3, 4], [5, 6]])

    regressor = Mock()
    regressor.predict.return_value = np.array(
        [[0.1, 0.5, 0.9], [0.2, 0.4, 0.8], [0.3, 0.6, 0.7]]
    )

    model.named_steps = {"transformations": transforms, "model": regressor}

    return model


@pytest.fixture
def mock_config():
    """Mock configuration."""
    with patch("stock_market_analytics.inference.inference_steps.config") as mock_cfg:
        mock_cfg.modeling.features = ["feature1", "feature2"]
        mock_cfg.modeling.quantiles = [0.1, 0.5, 0.9]
        yield mock_cfg


# -------------------------
# Test download_artifacts
# -------------------------


def test_download_artifacts_with_latest_version(mock_wandb_api):
    """Test download_artifacts with latest version."""
    with patch.dict(
        os.environ, {"WANDB_API_KEY": "test_key", "MODEL_NAME": "pipeline:latest"}
    ):
        model_dir, model_name = download_artifacts()

        assert model_dir == "/tmp/model_dir"
        assert model_name == "pipeline:latest"
        mock_wandb_api.artifact.assert_called_once_with(
            "san-cbtm/stock-market-analytics/pipeline:latest", type="model"
        )


def test_download_artifacts_with_specific_version(mock_wandb_api):
    """Test download_artifacts with specific version."""
    with (
        patch.dict(
            os.environ, {"WANDB_API_KEY": "test_key", "MODEL_NAME": "pipeline:v1.0"}
        ),
        patch("stock_market_analytics.inference.inference_steps.config") as mock_config,
    ):
        mock_config.wandb_key = "test_key"
        mock_config.model_name = "pipeline:v1.0"
        model_dir, model_name = download_artifacts()

        assert model_dir == "/tmp/model_dir"
        assert model_name == "pipeline:v1.0"
        mock_wandb_api.artifact.assert_called_once_with(
            "san-cbtm/stock-market-analytics/pipeline:v1.0", type="model"
        )


def test_download_artifacts_no_model_name(mock_wandb_api):
    """Test download_artifacts with no MODEL_NAME env var."""
    with patch.dict(os.environ, {"WANDB_API_KEY": "test_key"}, clear=True):
        model_dir, model_name = download_artifacts()

        assert model_dir == "/tmp/model_dir"
        assert model_name == "pipeline:latest"


def test_download_artifacts_no_version_in_name(mock_wandb_api):
    """Test download_artifacts with model name without version."""
    with (
        patch.dict(os.environ, {"WANDB_API_KEY": "test_key", "MODEL_NAME": "pipeline"}),
        patch("stock_market_analytics.inference.inference_steps.config") as mock_config,
    ):
        mock_config.wandb_key = "test_key"
        mock_config.model_name = "pipeline"
        model_dir, model_name = download_artifacts()

        assert model_dir == "/tmp/model_dir"
        assert model_name == "pipeline"


# -------------------------
# Test load_model
# -------------------------


def test_load_model(mock_joblib):
    """Test load_model function."""
    model_dir = "/tmp/model_dir"
    model_name = "pipeline:latest"

    result = load_model(model_dir, model_name)

    assert result == mock_joblib.return_value
    mock_joblib.assert_called_once_with("/tmp/model_dir/pipeline.pkl")


def test_load_model_with_version(mock_joblib):
    """Test load_model with versioned model name."""
    model_dir = "/tmp/model_dir"
    model_name = "pipeline:v1.0"

    result = load_model(model_dir, model_name)

    assert result == mock_joblib.return_value
    mock_joblib.assert_called_once_with("/tmp/model_dir/pipeline.pkl")


def test_load_model_no_version(mock_joblib):
    """Test load_model with model name without version."""
    model_dir = "/tmp/model_dir"
    model_name = "my_model"

    result = load_model(model_dir, model_name)

    assert result == mock_joblib.return_value
    mock_joblib.assert_called_once_with("/tmp/model_dir/my_model.pkl")


# -------------------------
# Test download_and_load_model
# -------------------------


def test_download_and_load_model(mock_wandb_api, mock_joblib):
    """Test combined download and load function."""
    with patch.dict(
        os.environ, {"WANDB_API_KEY": "test_key", "MODEL_NAME": "pipeline:latest"}
    ):
        result = download_and_load_model()

        assert result == mock_joblib.return_value
        mock_joblib.assert_called_once_with("/tmp/model_dir/pipeline.pkl")


# -------------------------
# Test get_inference_data
# -------------------------


@patch("stock_market_analytics.inference.inference_steps.generate_inference_features")
@patch("stock_market_analytics.inference.inference_steps.collect_inference_data")
def test_get_inference_data_success(mock_collect, mock_generate):
    """Test successful get_inference_data pipeline."""
    # Mock return values
    mock_raw_data = Mock()
    mock_collect.return_value = mock_raw_data

    mock_features = Mock()
    # Add shape attribute for print statement
    mock_features.shape = (2, 2)
    mock_features.to_pandas.return_value = pd.DataFrame(
        {"feature1": [1, 2], "feature2": [3, 4]}
    )
    mock_generate.return_value = mock_features

    result = get_inference_data("AAPL")

    # Check function calls
    mock_collect.assert_called_once_with("AAPL")
    mock_generate.assert_called_once_with(mock_raw_data)
    mock_features.to_pandas.assert_called_once()

    # Check result
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)


@patch("stock_market_analytics.inference.inference_steps.generate_inference_features")
@patch("stock_market_analytics.inference.inference_steps.collect_inference_data")
def test_get_inference_data_exception(mock_collect, mock_generate):
    """Test get_inference_data with exception."""
    mock_collect.side_effect = Exception("Data collection failed")

    with pytest.raises(Exception, match="Data collection failed"):
        get_inference_data("AAPL")


@patch("stock_market_analytics.inference.inference_steps.generate_inference_features")
@patch("stock_market_analytics.inference.inference_steps.collect_inference_data")
def test_get_inference_data_with_different_symbols(mock_collect, mock_generate):
    """Test get_inference_data with different stock symbols."""
    mock_raw_data = Mock()
    mock_collect.return_value = mock_raw_data

    mock_features = Mock()
    # Add shape attribute for print statement
    mock_features.shape = (1, 2)
    mock_features.to_pandas.return_value = pd.DataFrame(
        {"feature1": [1], "feature2": [2]}
    )
    mock_generate.return_value = mock_features

    # Test with different symbols
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        result = get_inference_data(symbol)
        mock_collect.assert_called_with(symbol)
        assert isinstance(result, pd.DataFrame)


# -------------------------
# Test make_prediction_intervals
# -------------------------


def test_make_prediction_intervals(mock_model, mock_config, sample_dataframe):
    """Test make_prediction_intervals function."""
    result = make_prediction_intervals(mock_model, sample_dataframe.copy())

    # Check model.predict was called with correct features
    mock_model.predict.assert_called_once()
    call_args = mock_model.predict.call_args[0][0]
    expected_features = sample_dataframe[mock_config.modeling.features]
    pd.testing.assert_frame_equal(call_args, expected_features)

    # Check result columns
    assert "pred_low_quantile" in result.columns
    assert "pred_high_quantile" in result.columns

    # Check predictions
    np.testing.assert_array_equal(result["pred_low_quantile"].values, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(result["pred_high_quantile"].values, [0.9, 0.8, 0.7])


def test_make_prediction_intervals_empty_dataframe(mock_model, mock_config):
    """Test make_prediction_intervals with empty DataFrame."""
    empty_df = pd.DataFrame(columns=["feature1", "feature2"])
    mock_model.predict.return_value = np.array([]).reshape(0, 2)

    result = make_prediction_intervals(mock_model, empty_df)

    assert "pred_low_quantile" in result.columns
    assert "pred_high_quantile" in result.columns
    assert len(result) == 0


# -------------------------
# Test predict_quantiles
# -------------------------


def test_predict_quantiles(mock_model, mock_config, sample_dataframe):
    """Test predict_quantiles function."""
    result = predict_quantiles(mock_model, sample_dataframe.copy())

    # Check transformations were called
    mock_model.named_steps["transformations"].transform.assert_called_once()

    # Check regressor predict was called
    mock_model.named_steps["model"].predict.assert_called_once_with(
        mock_model.named_steps["transformations"].transform.return_value,
        return_full_quantiles=True,
    )

    # Check quantile columns were created
    expected_cols = ["pred_Q_10", "pred_Q_50", "pred_Q_90"]
    for col in expected_cols:
        assert col in result.columns

    # Check predictions were assigned correctly
    np.testing.assert_array_equal(result["pred_Q_10"].values, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(result["pred_Q_50"].values, [0.5, 0.4, 0.6])
    np.testing.assert_array_equal(result["pred_Q_90"].values, [0.9, 0.8, 0.7])


def test_predict_quantiles_column_initialization(
    mock_model, mock_config, sample_dataframe
):
    """Test that quantile columns are properly initialized."""
    result = predict_quantiles(mock_model, sample_dataframe.copy())

    # Check that all quantile columns exist
    for q in mock_config.modeling.quantiles:
        quantile_col = f"pred_Q_{int(q * 100)}"
        assert quantile_col in result.columns


def test_predict_quantiles_feature_selection(mock_model, mock_config):
    """Test that predict_quantiles uses correct features."""
    # DataFrame with extra columns
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0],
            "feature2": [3.0, 4.0],
            "extra_col": [5.0, 6.0],
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        }
    )

    # Update mock to return correct array size (2 rows, 3 quantiles)
    mock_model.named_steps["model"].predict.return_value = np.array(
        [[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]]
    )

    result = predict_quantiles(mock_model, df)

    # Check that only configured features were used for transformation
    transform_call_args = mock_model.named_steps["transformations"].transform.call_args[
        0
    ][0]
    expected_features = df[mock_config.modeling.features]
    pd.testing.assert_frame_equal(transform_call_args, expected_features)


def test_predict_quantiles_empty_dataframe(mock_model, mock_config):
    """Test predict_quantiles with empty DataFrame."""
    empty_df = pd.DataFrame(columns=["feature1", "feature2"])

    # Mock empty predictions
    mock_model.named_steps["transformations"].transform.return_value = np.array(
        []
    ).reshape(0, 2)
    mock_model.named_steps["model"].predict.return_value = np.array([]).reshape(0, 3)

    result = predict_quantiles(mock_model, empty_df)

    # Check quantile columns exist but are empty
    expected_cols = ["pred_Q_10", "pred_Q_50", "pred_Q_90"]
    for col in expected_cols:
        assert col in result.columns
        assert len(result[col]) == 0

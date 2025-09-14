"""Unit tests for modeling step functions."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import date, datetime
from stock_market_analytics.config import config

from stock_market_analytics.modeling.modeling_steps import (
    load_features_data,
    prepare_modeling_data,
    get_adhoc_transforms_pipeline,
    get_catboost_multiquantile_model,
    analyze_feature_importance,
    get_calibrator,
    get_evaluator,
    get_baseline_model,
)


class TestLoadFeaturesData:
    """Test suite for load_features_data function."""

    def test_load_features_data_success(self, tmp_path):
        """Test successful features loading."""
        features_file = tmp_path / config.modeling.features_file
        test_data = pd.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2)],
                "symbol": ["AAPL", "AAPL"],
            }
        )

        for feature in config.modeling.features + [config.modeling.target]:
            test_data[feature] = np.random.randn(2)

        test_data.to_parquet(features_file)

        result = load_features_data(str(tmp_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_load_features_data_file_not_found(self, tmp_path):
        """Test file not found error."""
        with pytest.raises(FileNotFoundError):
            load_features_data(str(tmp_path / "nonexistent"))


class TestPrepareModelingData:
    """Test suite for prepare_modeling_data function."""

    def test_prepare_modeling_data_basic(self):
        """Test basic data preparation."""
        test_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100),
                "symbol": ["AAPL"] * 100,
            }
        )

        for feature in config.modeling.features + [config.modeling.target]:
            test_data[feature] = np.random.randn(100)

        result = prepare_modeling_data(df=test_data)

        assert isinstance(result, dict)
        # The actual return structure depends on the implementation
        # Just verify it returns a dictionary

class TestPipelineComponents:
    """Test suite for pipeline component creation functions."""

    def test_get_adhoc_transforms_pipeline(self):
        """Test adhoc transforms pipeline creation."""
        pipeline = get_adhoc_transforms_pipeline()

        assert pipeline is not None
        # Should be able to get pipeline without errors

    def test_get_catboost_multiquantile_model(self):
        """Test CatBoost model creation."""
        model = get_catboost_multiquantile_model()

        assert model is not None
        # Should have the expected protocol methods
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_get_catboost_model_with_params(self):
        """Test CatBoost model creation with custom params."""
        custom_params = {"learning_rate": 0.1, "depth": 3}
        model = get_catboost_multiquantile_model(params=custom_params)

        assert model is not None

    def test_get_calibrator(self):
        """Test calibrator creation."""
        calibrator = get_calibrator()

        assert calibrator is not None
        assert hasattr(calibrator, 'fit')
        assert hasattr(calibrator, 'transform')

    def test_get_evaluator(self):
        """Test evaluator creation."""
        evaluator = get_evaluator()

        assert evaluator is not None
        assert hasattr(evaluator, 'evaluate')

    def test_get_baseline_model(self):
        """Test baseline model creation."""
        model = get_baseline_model()

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')


class TestAnalyzeFeatureImportance:
    """Test suite for analyze_feature_importance function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_analyze_feature_importance_with_mock_model(self, mock_savefig, mock_show):
        """Test feature importance analysis with a mock model."""
        # Create a mock model with feature importance
        mock_model = Mock()
        mock_model.get_feature_importance.return_value = np.array([0.5, 0.3, 0.2])
        mock_model.feature_names_ = ["feature1", "feature2", "feature3"]

        # Should not raise an error
        analyze_feature_importance(mock_model)

        # Verify the model method was called
        mock_model.get_feature_importance.assert_called_once()

    def test_analyze_feature_importance_no_features(self):
        """Test feature importance with model that has no features."""
        mock_model = Mock()
        mock_model.get_feature_importance.return_value = np.array([])
        mock_model.feature_names_ = []

        # Should raise an error for empty feature list
        with pytest.raises(ValueError):
            analyze_feature_importance(mock_model)
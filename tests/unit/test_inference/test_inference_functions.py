"""Unit tests for inference functions."""

import pandas as pd
import polars as pl
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import date

from stock_market_analytics.inference.inference_functions import (
    create_inference_collection_plan,
    collect_inference_data,
    generate_inference_features,
)


class TestCreateInferenceCollectionPlan:
    """Test suite for create_inference_collection_plan function."""

    def test_create_inference_collection_plan_basic(self):
        """Test basic collection plan creation."""
        symbol = "AAPL"
        plan = create_inference_collection_plan(symbol)

        assert isinstance(plan, dict)
        assert plan["symbol"] == "AAPL"
        assert plan["period"] == "1y"

    def test_create_inference_collection_plan_lowercase_symbol(self):
        """Test collection plan with lowercase symbol."""
        symbol = "tsla"
        plan = create_inference_collection_plan(symbol)

        assert plan["symbol"] == "TSLA"  # Should be uppercased

    def test_create_inference_collection_plan_mixed_case_symbol(self):
        """Test collection plan with mixed case symbol."""
        symbol = "MsFt"
        plan = create_inference_collection_plan(symbol)

        assert plan["symbol"] == "MSFT"  # Should be uppercased


class TestCollectInferenceData:
    """Test suite for collect_inference_data function."""

    @patch('stock_market_analytics.inference.inference_functions.collect_and_process_symbol')
    @patch('builtins.print')  # Mock print to avoid output during tests
    def test_collect_inference_data_success(self, mock_print, mock_collect):
        """Test successful data collection."""
        # Mock successful collection result
        mock_data = pl.DataFrame({
            'date': [date(2023, 1, 1), date(2023, 1, 2)],
            'symbol': ['AAPL', 'AAPL'],
            'close': [150.0, 151.0],
            'volume': [1000, 1100]
        })

        mock_collect.return_value = {
            'data': mock_data,
            'new_metadata': {'status': 'success'}
        }

        result = collect_inference_data("AAPL")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert 'date' in result.columns
        assert 'symbol' in result.columns
        mock_collect.assert_called_once()

        # Check that the collection plan was created correctly
        call_args = mock_collect.call_args[0][0]
        assert call_args["symbol"] == "AAPL"
        assert call_args["period"] == "1y"

    @patch('stock_market_analytics.inference.inference_functions.collect_and_process_symbol')
    @patch('builtins.print')
    def test_collect_inference_data_failure(self, mock_print, mock_collect):
        """Test data collection failure."""
        # Mock failed collection result
        mock_collect.return_value = {
            'data': None,
            'new_metadata': {'status': 'failed_validation'}
        }

        with pytest.raises(RuntimeError, match="Data collection failed"):
            collect_inference_data("INVALID")

    @patch('stock_market_analytics.inference.inference_functions.collect_and_process_symbol')
    @patch('builtins.print')
    def test_collect_inference_data_no_metadata(self, mock_print, mock_collect):
        """Test data collection failure without metadata."""
        # Mock failed collection result without metadata
        mock_collect.return_value = {
            'data': None
        }

        with pytest.raises(RuntimeError, match="unknown_error"):
            collect_inference_data("TEST")


class TestGenerateInferenceFeatures:
    """Test suite for generate_inference_features function."""

    @patch('stock_market_analytics.inference.inference_functions.create_feature_pipeline')
    @patch('stock_market_analytics.inference.inference_functions.execute_feature_pipeline')
    @patch('builtins.print')
    def test_generate_inference_features_success(self, mock_print, mock_execute, mock_create):
        """Test successful feature generation."""
        # Mock input data
        raw_data = pl.DataFrame({
            'date': [date(2023, 1, 1), date(2023, 1, 2)],
            'symbol': ['AAPL', 'AAPL'],
            'close': [150.0, 151.0],
            'volume': [1000, 1100]
        })

        # Mock pipeline
        mock_pipeline = Mock()
        mock_create.return_value = mock_pipeline

        # Mock feature execution result
        mock_features = pl.DataFrame({
            'date': [date(2023, 1, 1), date(2023, 1, 2)],
            'symbol': ['AAPL', 'AAPL'],
            'feature1': [0.1, 0.2],
            'feature2': [0.5, 0.6]
        })
        mock_execute.return_value = mock_features

        result = generate_inference_features(raw_data)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns

        # Verify pipeline was created and executed
        mock_create.assert_called_once()
        mock_execute.assert_called_once()

        # Check execute_feature_pipeline was called with correct arguments
        call_args = mock_execute.call_args
        assert call_args[0][0] == mock_pipeline  # pipeline
        assert call_args[0][1].equals(raw_data)  # raw_data
        # Third argument should be features_config.as_dict

    @patch('stock_market_analytics.inference.inference_functions.create_feature_pipeline')
    @patch('stock_market_analytics.inference.inference_functions.execute_feature_pipeline')
    @patch('builtins.print')
    def test_generate_inference_features_pipeline_error(self, mock_print, mock_execute, mock_create):
        """Test feature generation with pipeline error."""
        raw_data = pl.DataFrame({
            'date': [date(2023, 1, 1)],
            'symbol': ['AAPL'],
            'close': [150.0]
        })

        # Mock pipeline creation success but execution failure
        mock_pipeline = Mock()
        mock_create.return_value = mock_pipeline
        mock_execute.side_effect = ValueError("Feature calculation failed")

        with pytest.raises(RuntimeError, match="Feature generation failed"):
            generate_inference_features(raw_data)

    @patch('stock_market_analytics.inference.inference_functions.create_feature_pipeline')
    @patch('builtins.print')
    def test_generate_inference_features_create_pipeline_error(self, mock_print, mock_create):
        """Test feature generation with pipeline creation error."""
        raw_data = pl.DataFrame({
            'date': [date(2023, 1, 1)],
            'symbol': ['AAPL'],
            'close': [150.0]
        })

        # Mock pipeline creation failure
        mock_create.side_effect = Exception("Pipeline creation failed")

        with pytest.raises(RuntimeError, match="Feature generation failed"):
            generate_inference_features(raw_data)

    def test_generate_inference_features_empty_data(self):
        """Test feature generation with empty data."""
        empty_data = pl.DataFrame()

        # This should either work with empty data or raise a clear error
        # The behavior depends on the implementation details
        with patch('stock_market_analytics.inference.inference_functions.create_feature_pipeline') as mock_create, \
             patch('stock_market_analytics.inference.inference_functions.execute_feature_pipeline') as mock_execute, \
             patch('builtins.print'):

            mock_pipeline = Mock()
            mock_create.return_value = mock_pipeline
            mock_execute.side_effect = Exception("Cannot process empty data")

            with pytest.raises(RuntimeError):
                generate_inference_features(empty_data)
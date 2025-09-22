"""
Integration tests for feature engineering workflow steps.

Tests the complete feature engineering workflow step functions, focusing on
end-to-end functionality while mocking external dependencies.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import pandas as pd
import polars as pl

from stock_market_analytics.feature_engineering import feature_steps


class TestFeatureEngineeringStepsIntegration:
    """Integration tests for feature engineering workflow steps."""

    @pytest.fixture
    def mock_environment(self):
        """Set up mock environment for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_stock_data(self):
        """Sample stock market data for feature engineering."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        data = []

        for symbol in ["AAPL", "GOOGL"]:
            for i, date in enumerate(dates):
                base_price = 150.0 if symbol == "AAPL" else 2800.0
                volatility = 0.02 + (i % 10) * 0.001

                data.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "open": base_price * (1 + volatility),
                        "high": base_price * (1 + volatility + 0.01),
                        "low": base_price * (1 - volatility),
                        "close": base_price * (1 + volatility - 0.005),
                        "volume": 1000000 + (i % 50) * 10000,
                        "adj_close": base_price * (1 + volatility - 0.005),
                    }
                )

        return pl.DataFrame(data)

    def test_load_stock_data_success(self, mock_environment, sample_stock_data):
        """Test successful stock data loading."""
        # Create historical data file with correct name
        historical_path = Path(mock_environment) / "stocks_history.parquet"
        sample_stock_data.write_parquet(historical_path)

        # Test loading
        result = feature_steps.load_stock_data(str(historical_path))

        assert not result.is_empty()
        assert "symbol" in result.columns
        assert "date" in result.columns

    def test_load_stock_data_missing_file(self, mock_environment):
        """Test loading stock data when file doesn't exist."""
        with pytest.raises(ValueError):
            feature_steps.load_stock_data(
                str(Path(mock_environment) / "stocks_history.parquet")
            )

    @patch(
        "stock_market_analytics.feature_engineering.feature_steps.execute_feature_pipeline"
    )
    @patch(
        "stock_market_analytics.feature_engineering.feature_steps.create_feature_pipeline"
    )
    def test_build_features_from_data_success(
        self,
        mock_create_pipeline,
        mock_execute_pipeline,
        mock_environment,
        sample_stock_data,
    ):
        """Test successful feature building."""
        # Create historical data file with correct name
        historical_path = Path(mock_environment) / "stocks_history.parquet"
        sample_stock_data.write_parquet(historical_path)

        # Mock pipeline creation and execution
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline

        features_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "GOOGL", "GOOGL"],
                "date": pd.date_range("2024-01-01", periods=4, freq="D"),
                "returns_1d": [0.001, 0.002, -0.001, 0.0015],
                "volatility_10d": [0.15, 0.16, 0.14, 0.155],
                "rsi_14d": [55, 60, 45, 50],
            }
        )
        mock_execute_pipeline.return_value = features_data

        # Test feature building
        stocks_path = Path(mock_environment) / "stocks_history.parquet"
        features_path = Path(mock_environment) / "stock_history_features.parquet"
        result = feature_steps.build_features_from_data(
            str(stocks_path), str(features_path)
        )

        assert result["status"] == "success"
        assert result["input_records"] == 200  # 100 days * 2 symbols
        assert result["output_records"] == 4
        assert "features_file" in result

    @patch(
        "stock_market_analytics.feature_engineering.feature_steps.execute_feature_pipeline"
    )
    @patch(
        "stock_market_analytics.feature_engineering.feature_steps.create_feature_pipeline"
    )
    def test_build_features_from_data_empty_result(
        self,
        mock_create_pipeline,
        mock_execute_pipeline,
        mock_environment,
        sample_stock_data,
    ):
        """Test feature building with empty result."""
        # Create historical data file with correct name
        historical_path = Path(mock_environment) / "stocks_history.parquet"
        sample_stock_data.write_parquet(historical_path)

        # Mock pipeline creation and execution
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline

        # Mock empty features result
        empty_features = pl.DataFrame()
        mock_execute_pipeline.return_value = empty_features

        # Test feature building
        stocks_path = Path(mock_environment) / "stocks_history.parquet"
        features_path = Path(mock_environment) / "stock_history_features.parquet"
        result = feature_steps.build_features_from_data(
            str(stocks_path), str(features_path)
        )

        assert result["status"] == "success"
        assert result["output_records"] == 0

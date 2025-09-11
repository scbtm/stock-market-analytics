"""Unit tests for feature engineering step functions."""

import pandas as pd
import polars as pl
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import date, datetime

from stock_market_analytics.feature_engineering.feature_steps import (
    load_stock_data,
    apply_time_filters,
    create_feature_pipeline,
    execute_feature_pipeline,
    save_features,
    build_features_from_data,
)


class TestLoadStockData:
    """Test suite for load_stock_data function."""

    def test_load_stock_data_success(self, tmp_path):
        """Test successful stock data loading."""
        # Create test parquet file
        stocks_file = tmp_path / "stocks_history.parquet"
        test_data = pl.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2)],
                "symbol": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [99.0, 100.0],
                "close": [102.0, 103.0],
                "volume": [1000000, 1100000],
            }
        )
        test_data.write_parquet(stocks_file)

        result = load_stock_data(tmp_path)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "symbol" in result.columns
        assert "date" in result.columns

    def test_load_stock_data_file_not_found(self, tmp_path):
        """Test FileNotFoundError when stocks file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Stocks history file not found"):
            load_stock_data(tmp_path)

    def test_load_stock_data_custom_filename(self, tmp_path):
        """Test loading with custom filename."""
        custom_file = tmp_path / "custom_stocks.parquet"
        test_data = pl.DataFrame(
            {"date": [date(2023, 1, 1)], "symbol": ["AAPL"], "close": [100.0]}
        )
        test_data.write_parquet(custom_file)

        result = load_stock_data(tmp_path, "custom_stocks.parquet")

        assert len(result) == 1
        assert result["symbol"][0] == "AAPL"

    def test_load_stock_data_corrupted_file(self, tmp_path):
        """Test ValueError when file is corrupted."""
        stocks_file = tmp_path / "stocks_history.parquet"
        stocks_file.write_text("corrupted parquet data")

        with pytest.raises(ValueError, match="Error loading stocks history file"):
            load_stock_data(tmp_path)


class TestApplyTimeFilters:
    """Test suite for apply_time_filters function."""

    def test_apply_time_filters_with_past_horizon(self):
        """Test time filtering with past horizon."""
        test_data = pl.DataFrame(
            {
                "date": [
                    date(2023, 1, 1),
                    date(2023, 6, 1),  # 151 days from start
                    date(2023, 12, 1),  # 334 days from start
                ],
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "close": [100.0, 110.0, 120.0],
            }
        )

        # Filter to last 200 days
        result = apply_time_filters(test_data, 200)

        # Should keep dates within 200 days of max date (2023-12-01)
        assert len(result) >= 1  # At least the most recent date
        # Should filter out very old dates
        max_date = result["date"].max()
        min_allowed_date = pd.Timestamp(max_date) - pd.Timedelta(days=200)
        assert all(pd.Timestamp(d) >= min_allowed_date for d in result["date"])

    def test_apply_time_filters_zero_horizon(self):
        """Test time filtering with zero past horizon."""
        test_data = pl.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 6, 1), date(2023, 12, 1)],
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "close": [100.0, 110.0, 120.0],
            }
        )

        result = apply_time_filters(test_data, 0)

        # Should return all data when horizon is 0
        assert len(result) == 3
        assert result.equals(test_data)

    def test_apply_time_filters_negative_horizon(self):
        """Test time filtering with negative past horizon."""
        test_data = pl.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 6, 1)],
                "symbol": ["AAPL", "AAPL"],
                "close": [100.0, 110.0],
            }
        )

        result = apply_time_filters(test_data, -10)

        # Should return all data when horizon is negative
        assert result.equals(test_data)

    def test_apply_time_filters_empty_data(self):
        """Test time filtering with empty dataframe."""
        empty_data = pl.DataFrame(
            schema={"date": pl.Date, "symbol": pl.Utf8, "close": pl.Float64}
        )

        result = apply_time_filters(empty_data, 30)

        assert len(result) == 0
        assert result.equals(empty_data)


class TestCreateFeaturePipeline:
    """Test suite for create_feature_pipeline function."""

    @patch("stock_market_analytics.feature_engineering.feature_steps.driver")
    def test_create_feature_pipeline(self, mock_driver):
        """Test feature pipeline creation."""
        mock_builder = Mock()
        mock_driver_instance = Mock()
        mock_builder.with_modules.return_value = mock_builder
        mock_builder.build.return_value = mock_driver_instance
        mock_driver.Builder.return_value = mock_builder

        result = create_feature_pipeline()

        assert result == mock_driver_instance
        mock_driver.Builder.assert_called_once()
        mock_builder.with_modules.assert_called_once()
        mock_builder.build.assert_called_once()


class TestExecuteFeaturePipeline:
    """Test suite for execute_feature_pipeline function."""

    def test_execute_feature_pipeline(self):
        """Test feature pipeline execution."""
        # Create mock driver
        mock_driver = Mock()
        mock_results = {"df_features": pl.DataFrame({"feature1": [1, 2, 3]})}
        mock_driver.execute.return_value = mock_results

        # Create test data
        raw_data = pl.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "close": [100.0, 101.0, 102.0],
            }
        )

        config_dict = {"param1": "value1", "param2": 42}

        result = execute_feature_pipeline(mock_driver, raw_data, config_dict)

        assert isinstance(result, pl.DataFrame)
        assert "feature1" in result.columns
        mock_driver.execute.assert_called_once_with(
            final_vars=["df_features"], inputs={"raw_df": raw_data, **config_dict}
        )


class TestSaveFeatures:
    """Test suite for save_features function."""

    def test_save_features_default_filename(self, tmp_path):
        """Test saving features with default filename."""
        features_data = pl.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2)],
                "symbol": ["AAPL", "AAPL"],
                "feature1": [0.5, 0.6],
                "feature2": [1.2, 1.3],
            }
        )

        save_features(features_data, tmp_path)

        features_file = tmp_path / "stock_history_features.parquet"
        assert features_file.exists()

        # Verify saved data
        saved_data = pl.read_parquet(features_file)
        assert saved_data.equals(features_data)

    def test_save_features_custom_filename(self, tmp_path):
        """Test saving features with custom filename."""
        features_data = pl.DataFrame({"symbol": ["AAPL"], "feature1": [0.5]})

        save_features(features_data, tmp_path, "custom_features.parquet")

        features_file = tmp_path / "custom_features.parquet"
        assert features_file.exists()


class TestBuildFeaturesFromData:
    """Test suite for build_features_from_data function."""

    def test_build_features_from_data_success(self, tmp_path):
        """Test successful feature building workflow."""
        # Create test stock data
        stocks_file = tmp_path / "stocks_history.parquet"
        test_data = pl.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [102.0, 103.0, 104.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )
        test_data.write_parquet(stocks_file)

        # Mock the feature pipeline
        with (
            patch(
                "stock_market_analytics.feature_engineering.feature_steps.create_feature_pipeline"
            ) as mock_create,
            patch(
                "stock_market_analytics.feature_engineering.feature_steps.execute_feature_pipeline"
            ) as mock_execute,
            patch(
                "stock_market_analytics.feature_engineering.feature_steps.features_config"
            ) as mock_config,
        ):
            # Setup mocks
            mock_pipeline = Mock()
            mock_create.return_value = mock_pipeline

            mock_features = pl.DataFrame(
                {
                    "date": [date(2023, 1, 1), date(2023, 1, 2)],
                    "symbol": ["AAPL", "AAPL"],
                    "feature1": [0.5, 0.6],
                    "feature2": [1.2, 1.3],
                }
            )
            mock_execute.return_value = mock_features

            mock_config.past_horizon = 30
            mock_config.as_dict = {"param1": "value1"}

            result = build_features_from_data(tmp_path)

            # Verify result
            assert result["status"] == "success"
            assert result["input_records"] == 3  # All records in filtered data
            assert result["output_records"] == 2  # Features output
            assert result["features_file"] == "stock_history_features.parquet"

            # Verify pipeline was called correctly
            mock_create.assert_called_once()
            mock_execute.assert_called_once()

            # Verify features file was created
            features_file = tmp_path / "stock_history_features.parquet"
            assert features_file.exists()

    def test_build_features_from_data_custom_files(self, tmp_path):
        """Test feature building with custom file names."""
        # Create test stock data with custom name
        custom_stocks_file = tmp_path / "custom_stocks.parquet"
        test_data = pl.DataFrame(
            {
                "date": [date(2023, 1, 1)],
                "symbol": ["AAPL"],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [102.0],
                "volume": [1000000],
            }
        )
        test_data.write_parquet(custom_stocks_file)

        # Mock the feature pipeline
        with (
            patch(
                "stock_market_analytics.feature_engineering.feature_steps.create_feature_pipeline"
            ) as mock_create,
            patch(
                "stock_market_analytics.feature_engineering.feature_steps.execute_feature_pipeline"
            ) as mock_execute,
            patch(
                "stock_market_analytics.feature_engineering.feature_steps.features_config"
            ) as mock_config,
        ):
            mock_pipeline = Mock()
            mock_create.return_value = mock_pipeline

            mock_features = pl.DataFrame({"symbol": ["AAPL"], "feature1": [0.5]})
            mock_execute.return_value = mock_features

            mock_config.past_horizon = 0  # No time filtering
            mock_config.as_dict = {}

            result = build_features_from_data(
                tmp_path,
                stocks_history_file="custom_stocks.parquet",
                features_file="custom_features.parquet",
            )

            assert result["status"] == "success"
            assert result["features_file"] == "custom_features.parquet"

            # Verify custom features file was created
            features_file = tmp_path / "custom_features.parquet"
            assert features_file.exists()

    def test_build_features_from_data_missing_stock_file(self, tmp_path):
        """Test feature building when stock file is missing."""
        with pytest.raises(FileNotFoundError):
            build_features_from_data(tmp_path)

    def test_build_features_from_data_empty_input(self, tmp_path):
        """Test feature building with empty input data."""
        # Create empty stock data
        stocks_file = tmp_path / "stocks_history.parquet"
        empty_data = pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )
        empty_data.write_parquet(stocks_file)

        # Mock the feature pipeline
        with (
            patch(
                "stock_market_analytics.feature_engineering.feature_steps.create_feature_pipeline"
            ) as mock_create,
            patch(
                "stock_market_analytics.feature_engineering.feature_steps.execute_feature_pipeline"
            ) as mock_execute,
            patch(
                "stock_market_analytics.feature_engineering.feature_steps.features_config"
            ) as mock_config,
        ):
            mock_pipeline = Mock()
            mock_create.return_value = mock_pipeline

            # Empty features output
            empty_features = pl.DataFrame(
                schema={"date": pl.Date, "symbol": pl.Utf8, "feature1": pl.Float64}
            )
            mock_execute.return_value = empty_features

            mock_config.past_horizon = 30
            mock_config.as_dict = {}

            result = build_features_from_data(tmp_path)

            assert result["status"] == "success"
            assert result["input_records"] == 0
            assert result["output_records"] == 0

    @patch(
        "stock_market_analytics.feature_engineering.feature_steps.create_feature_pipeline"
    )
    def test_build_features_from_data_pipeline_error(self, mock_create, tmp_path):
        """Test feature building when pipeline creation fails."""
        # Create test stock data
        stocks_file = tmp_path / "stocks_history.parquet"
        test_data = pl.DataFrame(
            {"date": [date(2023, 1, 1)], "symbol": ["AAPL"], "close": [100.0]}
        )
        test_data.write_parquet(stocks_file)

        # Make pipeline creation fail
        mock_create.side_effect = Exception("Pipeline creation failed")

        with patch(
            "stock_market_analytics.feature_engineering.feature_steps.features_config"
        ) as mock_config:
            mock_config.past_horizon = 30

            with pytest.raises(Exception, match="Pipeline creation failed"):
                build_features_from_data(tmp_path)

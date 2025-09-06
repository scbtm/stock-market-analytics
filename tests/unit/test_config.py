"""Simple unit tests for configuration module."""

import os
from pathlib import Path

import pytest

from stock_market_analytics.config import (
    AppConfig,
    DataCollectionConfig,
    FeatureEngineeringConfig,
    ModelingConfig,
)


class TestDataCollectionConfig:
    """Test suite for DataCollectionConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DataCollectionConfig()
        
        assert config.tickers_file == "tickers.csv"
        assert config.metadata_file == "metadata.csv"
        assert config.stocks_history_file == "stocks_history.parquet"
        assert "Symbol" in config.required_ticker_columns

    def test_ticker_column_mapping(self):
        """Test ticker column mapping generation."""
        config = DataCollectionConfig()
        mapping = config.ticker_column_mapping
        
        assert mapping["Symbol"] == "symbol"
        assert mapping["IPO Year"] == "ipo_year"


class TestFeatureEngineeringConfig:
    """Test suite for FeatureEngineeringConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = FeatureEngineeringConfig()
        
        assert config.horizon == 5
        assert config.past_horizon == 7 * 280

    def test_window_properties(self):
        """Test calculated window properties."""
        config = FeatureEngineeringConfig()
        
        assert config.short_window == config.horizon * 3
        assert config.long_window == config.horizon * 5

    def test_ichimoku_params(self):
        """Test ichimoku parameter generation."""
        config = FeatureEngineeringConfig()
        params = config.ichimoku_params
        
        assert "p1" in params
        assert params["p1"] == config.horizon * 2


class TestModelingConfig:
    """Test suite for ModelingConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelingConfig()
        
        assert config.features_file == "stock_history_features.parquet"
        assert config.target == "y_log_returns"
        assert config.target_coverage == 0.8

    def test_quantile_indices(self):
        """Test quantile indices mapping."""
        config = ModelingConfig()
        indices = config.quantile_indices
        
        assert indices["LOW"] == 0
        assert indices["MID"] == 2
        assert indices["HIGH"] == 4

    def test_has_required_features(self):
        """Test that required features are present."""
        config = ModelingConfig()
        
        assert "log_returns_d" in config.features
        assert "rsi_ewm" in config.features
        assert len(config.features) > 10


class TestAppConfig:
    """Test suite for AppConfig."""
    
    def test_default_initialization(self):
        """Test default app config initialization."""
        config = AppConfig()
        
        assert isinstance(config.data_collection, DataCollectionConfig)
        assert isinstance(config.feature_engineering, FeatureEngineeringConfig)
        assert isinstance(config.modeling, ModelingConfig)

    def test_environment_variable_loading(self):
        """Test loading from environment variables."""
        # Set environment variables
        os.environ["BASE_DATA_PATH"] = "/tmp/test_data"
        os.environ["WANDB_KEY"] = "test_key"
        
        config = AppConfig()
        
        assert config.base_data_path == Path("/tmp/test_data")
        assert config.wandb_key == "test_key"
        
        # Clean up
        del os.environ["BASE_DATA_PATH"]
        del os.environ["WANDB_KEY"]
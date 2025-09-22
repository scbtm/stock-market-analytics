"""
Integration tests for data collection workflow steps.

Tests the complete data collection workflow step functions, focusing on
end-to-end functionality while mocking external dependencies.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import pandas as pd
import polars as pl

from stock_market_analytics.data_collection import collection_steps


class TestDataCollectionStepsIntegration:
    """Integration tests for data collection workflow steps."""

    @pytest.fixture
    def mock_environment(self):
        """Set up mock environment for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_tickers_data(self):
        """Sample ticker data."""
        return pd.DataFrame(
            {
                "Symbol": ["AAPL", "GOOGL", "MSFT"],
                "Name": ["Apple Inc.", "Alphabet Inc.", "Microsoft Corp."],
                "Country": ["USA", "USA", "USA"],
                "IPO Year": [1980, 2004, 1986],
                "Sector": ["Technology", "Technology", "Technology"],
                "Industry": ["Consumer Electronics", "Internet Content", "Software"],
            }
        )

    @pytest.fixture
    def sample_stock_data(self):
        """Sample stock market data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = []

        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            for date in dates:
                data.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "open": 150.0 + (hash(f"{symbol}{date}") % 10),
                        "high": 155.0 + (hash(f"{symbol}{date}") % 10),
                        "low": 148.0 + (hash(f"{symbol}{date}") % 10),
                        "close": 152.0 + (hash(f"{symbol}{date}") % 10),
                        "volume": 1000000 + (hash(f"{symbol}{date}") % 100000),
                    }
                )

        return pl.DataFrame(data)

    def test_load_tickers_success(self, mock_environment, sample_tickers_data):
        """Test successful ticker loading."""
        # Create tickers file with correct name from config
        tickers_path = Path(mock_environment) / "top_200_tickers.csv"
        sample_tickers_data.to_csv(tickers_path, index=False)

        # Test loading
        result = collection_steps.load_tickers(str(tickers_path))

        assert len(result) == 3
        assert result[0]["symbol"] == "AAPL"
        assert result[1]["symbol"] == "GOOGL"
        assert result[2]["symbol"] == "MSFT"

    def test_load_tickers_missing_file(self, mock_environment):
        """Test loading tickers when file doesn't exist."""
        with pytest.raises(ValueError):
            collection_steps.load_tickers(
                str(Path(mock_environment) / "top_200_tickers.csv")
            )

    def test_load_metadata_success(self, mock_environment):
        """Test successful metadata loading."""
        # Create metadata file with correct column names from config
        metadata_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL"],
                "last_ingestion": ["2024-01-01", "2024-01-01"],
                "max_date_recorded": ["2024-01-01", "2024-01-01"],
                "status": ["success", "success"],
            }
        )

        metadata_path = Path(mock_environment) / "metadata.csv"
        metadata_data.to_csv(metadata_path, index=False)

        # Test loading
        result = collection_steps.load_metadata(str(metadata_path))

        assert result is not None
        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[1]["symbol"] == "GOOGL"

    def test_load_metadata_missing_file(self, mock_environment):
        """Test loading metadata when file doesn't exist."""
        result = collection_steps.load_metadata(
            str(Path(mock_environment) / "metadata.csv")
        )
        assert result == []

    def test_create_collection_plans(self, mock_environment, sample_tickers_data):
        """Test creation of collection plans."""
        # Create tickers file and load it properly to get the right format
        tickers_path = Path(mock_environment) / "top_200_tickers.csv"
        sample_tickers_data.to_csv(tickers_path, index=False)

        # Load tickers using the actual function to get proper format
        tickers_list = collection_steps.load_tickers(str(tickers_path))

        # Test with no existing metadata (full collection)
        plans = collection_steps.create_collection_plans(tickers_list, [])

        assert len(plans) == 3
        for plan in plans:
            assert "symbol" in plan
            assert "period" in plan
            assert plan["symbol"] in ["AAPL", "GOOGL", "MSFT"]

    @patch("stock_market_analytics.data_collection.collection_steps.YFinanceCollector")
    @patch(
        "stock_market_analytics.data_collection.collection_steps.ContinuousTimelineProcessor"
    )
    def test_collect_and_process_symbol_success(
        self, mock_processor_class, mock_collector_class, sample_stock_data
    ):
        """Test successful symbol collection and processing."""
        # Mock collector
        mock_collector = Mock()
        filtered_data = sample_stock_data.filter(pl.col("symbol") == "AAPL")
        mock_collector.get_historical_data.return_value = filtered_data
        mock_collector.collection_successful = True
        mock_collector_class.return_value = mock_collector

        # Mock processor
        mock_processor = Mock()
        mock_processor.process.return_value = filtered_data
        mock_processor.processing_successful = True
        mock_processor_class.return_value = mock_processor

        # Create a collection plan dictionary
        collection_plan = {"symbol": "AAPL", "period": "max", "interval": "1d"}

        # Test collection
        result = collection_steps.collect_and_process_symbol(collection_plan)

        assert result["data"] is not None
        assert result["new_metadata"]["symbol"] == "AAPL"
        assert result["new_metadata"]["status"] == "active"

    @patch("stock_market_analytics.data_collection.collection_steps.YFinanceCollector")
    def test_collect_and_process_symbol_failure(self, mock_collector_class):
        """Test handling of collection failures."""
        # Mock collector failure
        mock_collector = Mock()
        mock_collector.collect.side_effect = Exception("API timeout")
        mock_collector_class.return_value = mock_collector

        # Create a collection plan dictionary
        collection_plan = {"symbol": "AAPL", "period": "max", "interval": "1d"}

        # Test collection
        result = collection_steps.collect_and_process_symbol(collection_plan)

        assert result["data"] is None
        assert result["new_metadata"]["symbol"] == "AAPL"
        assert result["new_metadata"]["status"] == "collection_issue"

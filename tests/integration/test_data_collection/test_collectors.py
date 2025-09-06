"""Integration tests for data collectors - testing external API interactions with mocks."""

from unittest.mock import Mock, patch

import pandas as pd
import polars as pl
import pytest

from stock_market_analytics.data_collection.collectors import YFinanceCollector
from stock_market_analytics.data_collection.models import YFinanceCollectionPlan


class TestYFinanceCollectorIntegration:
    """Integration tests for YFinanceCollector external API interactions."""

    @pytest.fixture
    def sample_yfinance_data(self):
        """Create sample yfinance pandas DataFrame output."""
        dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
        data = {
            "Open": [150.0, 151.0, 152.0, 153.0, 154.0],
            "High": [151.0, 152.0, 153.0, 154.0, 155.0],
            "Low": [149.0, 150.0, 151.0, 152.0, 153.0],
            "Close": [150.5, 151.5, 152.5, 153.5, 154.5],
            "Volume": [100_000, 110_000, 120_000, 130_000, 140_000],
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = "Date"
        return df

    @pytest.fixture
    def empty_yfinance_data(self):
        """Create empty yfinance pandas DataFrame output."""
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    @patch("stock_market_analytics.data_collection.collectors.yfinance_collector.yf")
    def test_get_historical_data_success(self, mock_yf, sample_yfinance_data):
        """Test successful data collection with valid yfinance response."""
        # Mock the Ticker and its history method
        mock_ticker = Mock()
        mock_ticker.history.return_value = sample_yfinance_data
        mock_yf.Ticker.return_value = mock_ticker

        collection_plan = YFinanceCollectionPlan(symbol="AAPL", period="1y")
        collector = YFinanceCollector(collection_plan=collection_plan)

        result = collector.get_historical_data()

        # Verify the result is a Polars DataFrame with correct schema
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (5, 7)  # 5 rows, 7 columns
        assert list(result.columns) == [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        # Verify data types
        assert result["date"].dtype == pl.Date
        assert result["symbol"].dtype == pl.Utf8
        assert result["open"].dtype == pl.Float64
        assert result["high"].dtype == pl.Float64
        assert result["low"].dtype == pl.Float64
        assert result["close"].dtype == pl.Float64
        assert result["volume"].dtype == pl.Int64

        # Verify the data content
        assert result["symbol"].to_list() == ["AAPL"] * 5
        assert result["open"].to_list() == [150.0, 151.0, 152.0, 153.0, 154.0]
        assert result["close"].to_list() == [150.5, 151.5, 152.5, 153.5, 154.5]

        # Verify collection status
        assert collector.collection_successful is True
        assert collector.collected_empty_data is False
        assert collector.errors_during_collection is False

        # Verify yfinance was called correctly
        mock_yf.Ticker.assert_called_once_with("AAPL")
        mock_ticker.history.assert_called_once_with(period="1y")

    @patch("stock_market_analytics.data_collection.collectors.yfinance_collector.yf")
    def test_get_historical_data_empty_response(self, mock_yf, empty_yfinance_data):
        """Test data collection when yfinance returns empty data."""
        # Mock the Ticker and its history method
        mock_ticker = Mock()
        mock_ticker.history.return_value = empty_yfinance_data
        mock_yf.Ticker.return_value = mock_ticker

        collection_plan = YFinanceCollectionPlan(symbol="INVALID", period="1y")
        collector = YFinanceCollector(collection_plan=collection_plan)

        result = collector.get_historical_data()

        # Verify the result is an empty Polars DataFrame with correct schema
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (0, 7)  # 0 rows, 7 columns
        assert list(result.columns) == [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        # Verify data types
        assert result["date"].dtype == pl.Date
        assert result["symbol"].dtype == pl.Utf8
        assert result["open"].dtype == pl.Float64
        assert result["high"].dtype == pl.Float64
        assert result["low"].dtype == pl.Float64
        assert result["close"].dtype == pl.Float64
        assert result["volume"].dtype == pl.Int64

        assert result.is_empty()

        # Verify collection status
        assert collector.collection_successful is False
        assert collector.collected_empty_data is True
        assert collector.errors_during_collection is False

    @patch("stock_market_analytics.data_collection.collectors.yfinance_collector.yf")
    def test_get_historical_data_exception_handling(self, mock_yf):
        """Test data collection when yfinance raises an exception."""
        # Mock the Ticker to raise an exception
        mock_yf.Ticker.side_effect = Exception("Network error")

        collection_plan = YFinanceCollectionPlan(symbol="AAPL", period="1y")
        collector = YFinanceCollector(collection_plan=collection_plan)

        result = collector.get_historical_data()

        # Verify the result is an empty Polars DataFrame with correct schema
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (0, 7)  # 0 rows, 7 columns
        assert list(result.columns) == [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        # Verify collection status
        assert collector.collection_successful is False
        assert collector.collected_empty_data is False
        assert collector.errors_during_collection is True
        assert collector.error_message == "Network error"

    @patch("stock_market_analytics.data_collection.collectors.yfinance_collector.yf")
    def test_get_historical_data_with_start_end_dates(
        self, mock_yf, sample_yfinance_data
    ):
        """Test data collection with start and end dates."""
        # Mock the Ticker and its history method
        mock_ticker = Mock()
        mock_ticker.history.return_value = sample_yfinance_data
        mock_yf.Ticker.return_value = mock_ticker

        collection_plan = YFinanceCollectionPlan(
            symbol="AAPL", start="2023-01-01", end="2023-12-31"
        )
        collector = YFinanceCollector(collection_plan=collection_plan)

        result = collector.get_historical_data()

        # Verify the result
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (5, 7)

        # Verify yfinance was called with correct parameters
        mock_yf.Ticker.assert_called_once_with("AAPL")
        mock_ticker.history.assert_called_once_with(
            start="2023-01-01", end="2023-12-31"
        )

    @patch("stock_market_analytics.data_collection.collectors.yfinance_collector.yf")
    def test_get_historical_data_with_interval(self, mock_yf, sample_yfinance_data):
        """Test data collection with interval parameter."""
        # Mock the Ticker and its history method
        mock_ticker = Mock()
        mock_ticker.history.return_value = sample_yfinance_data
        mock_yf.Ticker.return_value = mock_ticker

        collection_plan = YFinanceCollectionPlan(
            symbol="AAPL", period="1mo", interval="1h"
        )
        collector = YFinanceCollector(collection_plan=collection_plan)

        result = collector.get_historical_data()

        # Verify the result
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (5, 7)

        # Verify yfinance was called with correct parameters
        mock_yf.Ticker.assert_called_once_with("AAPL")
        mock_ticker.history.assert_called_once_with(period="1mo", interval="1h")

    @patch("stock_market_analytics.data_collection.collectors.yfinance_collector.yf")
    def test_get_historical_data_column_mapping(self, mock_yf):
        """Test that column mapping works correctly with different yfinance output."""
        # Create custom yfinance data with different column names
        dates = pd.date_range(start="2023-01-01", end="2023-01-03", freq="D")
        data = {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [500000, 600000, 700000],
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = "Date"

        # Mock the Ticker and its history method
        mock_ticker = Mock()
        mock_ticker.history.return_value = df
        mock_yf.Ticker.return_value = mock_ticker

        collection_plan = YFinanceCollectionPlan(symbol="TSLA", period="1y")
        collector = YFinanceCollector(collection_plan=collection_plan)

        result = collector.get_historical_data()

        # Verify the result has correct column names and data
        assert list(result.columns) == [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert result["symbol"].to_list() == ["TSLA"] * 3
        assert result["open"].to_list() == [100.0, 101.0, 102.0]
        assert result["close"].to_list() == [100.5, 101.5, 102.5]
        assert result["volume"].to_list() == [500000, 600000, 700000]

    @patch("stock_market_analytics.data_collection.collectors.yfinance_collector.yf")
    def test_collector_state_management(self, mock_yf, sample_yfinance_data):
        """Test that collector state is properly managed across multiple calls."""
        # Mock the Ticker and its history method
        mock_ticker = Mock()
        mock_ticker.history.return_value = sample_yfinance_data
        mock_yf.Ticker.return_value = mock_ticker

        collection_plan = YFinanceCollectionPlan(symbol="AAPL", period="1y")
        collector = YFinanceCollector(collection_plan=collection_plan)

        # Initial state
        assert collector.collection_successful is False
        assert collector.collected_empty_data is False
        assert collector.errors_during_collection is False

        # First successful call
        result1 = collector.get_historical_data()
        assert collector.collection_successful is True
        assert collector.collected_empty_data is False
        assert collector.errors_during_collection is False
        assert result1.shape == (5, 7)

        # Second call should reset state and succeed again
        result2 = collector.get_historical_data()
        assert collector.collection_successful is True
        assert collector.collected_empty_data is False
        assert collector.errors_during_collection is False
        assert result2.shape == (5, 7)

        # Verify yfinance was called twice
        assert mock_yf.Ticker.call_count == 2
        assert mock_ticker.history.call_count == 2
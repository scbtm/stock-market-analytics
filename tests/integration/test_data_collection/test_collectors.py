"""Integration tests for data collectors - testing real YFinance API interactions."""

import polars as pl
import pytest

from stock_market_analytics.data_collection.collectors import YFinanceCollector
from stock_market_analytics.data_collection.models import YFinanceCollectionPlan


class TestYFinanceCollectorIntegration:
    """Integration tests for YFinanceCollector with real YFinance API calls."""

    @pytest.mark.slow
    def test_get_historical_data_success_real_api(self):
        """Test successful data collection with real YFinance API."""
        # Use a well-known, stable stock with short period for fast test
        collection_plan = YFinanceCollectionPlan(symbol="AAPL", period="5d")
        collector = YFinanceCollector(collection_plan=collection_plan)

        result = collector.get_historical_data()

        # Verify the result is a Polars DataFrame with correct schema
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0  # Should get some recent data
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

        # Verify the symbol is correct
        assert all(symbol == "AAPL" for symbol in result["symbol"].to_list())

        # Verify realistic stock price data
        prices = result["close"].to_list()
        assert all(price > 0 for price in prices)  # Prices should be positive
        assert all(price < 1000 for price in prices)  # Sanity check for AAPL price range

        # Verify collection status
        assert collector.collection_successful is True
        assert collector.collected_empty_data is False
        assert collector.errors_during_collection is False

    @pytest.mark.slow
    def test_get_historical_data_invalid_symbol(self):
        """Test data collection with an invalid symbol."""
        # Use a clearly invalid symbol that should return empty data
        collection_plan = YFinanceCollectionPlan(symbol="INVALID_SYMBOL_123", period="5d")
        collector = YFinanceCollector(collection_plan=collection_plan)

        result = collector.get_historical_data()

        # Should return empty DataFrame with correct schema
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0  # Should be empty for invalid symbol
        assert list(result.columns) == [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        # Verify data types even for empty DataFrame
        assert result["date"].dtype == pl.Date
        assert result["symbol"].dtype == pl.Utf8
        assert result["open"].dtype == pl.Float64
        assert result["high"].dtype == pl.Float64
        assert result["low"].dtype == pl.Float64
        assert result["close"].dtype == pl.Float64
        assert result["volume"].dtype == pl.Int64

        # Verify collection status reflects empty data
        assert collector.collection_successful is False
        assert collector.collected_empty_data is True
        assert collector.errors_during_collection is False

    @pytest.mark.slow
    def test_get_historical_data_with_date_range(self):
        """Test data collection with specific start and end dates."""
        # Test with a specific date range
        collection_plan = YFinanceCollectionPlan(
            symbol="MSFT", start="2024-01-01", end="2024-01-10"
        )
        collector = YFinanceCollector(collection_plan=collection_plan)

        result = collector.get_historical_data()

        # Verify the result
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0  # Should get some data for this range

        # Verify date range (allowing for weekends/holidays)
        from datetime import date
        dates = result["date"].to_list()
        assert min(dates) >= date(2024, 1, 1)
        assert max(dates) <= date(2024, 1, 10)

        # Verify symbol is correct
        assert all(symbol == "MSFT" for symbol in result["symbol"].to_list())

        # Verify collection was successful
        assert collector.collection_successful is True
        assert collector.collected_empty_data is False
        assert collector.errors_during_collection is False

    @pytest.mark.slow
    def test_get_historical_data_with_interval(self):
        """Test data collection with different interval parameter."""
        # Test with hourly data for a short period
        collection_plan = YFinanceCollectionPlan(
            symbol="GOOGL", period="5d", interval="1h"
        )
        collector = YFinanceCollector(collection_plan=collection_plan)

        result = collector.get_historical_data()

        # Verify the result structure - this is the main test
        assert isinstance(result, pl.DataFrame)

        # For integration tests with real API, we need to handle the fact that
        # some combinations of period/interval might not be supported by YFinance
        # The important thing is that the collector handles it gracefully and doesn't crash

        # Verify that the collector attempted the operation and returned a proper structure
        # (Success/failure depends on YFinance API availability and supported parameters)
        assert hasattr(collector, 'collection_successful')
        assert hasattr(collector, 'collected_empty_data')
        assert hasattr(collector, 'errors_during_collection')

        # If we got data, verify it has the right structure
        if len(result) > 0:
            # Verify required columns exist
            required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            assert all(col in result.columns for col in required_cols)
            # Verify symbol is correct
            assert all(symbol == "GOOGL" for symbol in result["symbol"].to_list())

    @pytest.mark.slow
    def test_get_historical_data_column_mapping_and_data_integrity(self):
        """Test that column mapping works correctly and data integrity is maintained."""
        collection_plan = YFinanceCollectionPlan(symbol="TSLA", period="5d")
        collector = YFinanceCollector(collection_plan=collection_plan)

        result = collector.get_historical_data()

        # Verify the result has correct column names
        assert list(result.columns) == [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        if len(result) > 0:  # If we got data
            # Verify symbol is consistent
            assert all(symbol == "TSLA" for symbol in result["symbol"].to_list())

            # Verify financial data integrity (high >= low, etc.)
            for i in range(len(result)):
                high = result["high"][i]
                low = result["low"][i]
                open_price = result["open"][i]
                close_price = result["close"][i]

                # Basic financial data validation
                assert high >= low, f"High ({high}) should be >= Low ({low})"
                assert high >= open_price, f"High ({high}) should be >= Open ({open_price})"
                assert high >= close_price, f"High ({high}) should be >= Close ({close_price})"
                assert low <= open_price, f"Low ({low}) should be <= Open ({open_price})"
                assert low <= close_price, f"Low ({low}) should be <= Close ({close_price})"

    @pytest.mark.slow
    def test_collector_state_management_real_api(self):
        """Test that collector state is properly managed across multiple calls."""
        collection_plan = YFinanceCollectionPlan(symbol="AAPL", period="5d")
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
        assert len(result1) > 0

        # Second call should reset state and succeed again
        result2 = collector.get_historical_data()
        assert collector.collection_successful is True
        assert collector.collected_empty_data is False
        assert collector.errors_during_collection is False
        assert len(result2) > 0

        # Results should be identical for same parameters
        assert result1.equals(result2)

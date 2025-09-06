"""Unit tests for data collectors - testing initialization and basic behavior only."""

import inspect

import pytest

from stock_market_analytics.data_collection.collectors import YFinanceCollector
from stock_market_analytics.data_collection.models import YFinanceCollectionPlan


class TestYFinanceCollectorUnit:
    """Pure unit tests for YFinanceCollector initialization and basic functionality."""

    def test_yfinance_collector_initialization_with_collection_plan(self):
        """Test YFinanceCollector initialization with a collection plan."""
        collection_plan = YFinanceCollectionPlan(symbol="AAPL", period="1y")
        collector = YFinanceCollector(collection_plan=collection_plan)

        assert collector.collection_plan == collection_plan
        assert collector.collection_plan.symbol == "AAPL"
        assert collector.collection_plan.period == "1y"
        assert collector.yf_params == {"period": "1y"}
        assert collector.collection_successful is False
        assert collector.collected_empty_data is False
        assert collector.errors_during_collection is False

    def test_yfinance_collector_initialization_with_kwargs(self):
        """Test YFinanceCollector initialization with kwargs."""
        collector = YFinanceCollector(symbol="AAPL", period="1y")

        assert collector.collection_plan.symbol == "AAPL"
        assert collector.collection_plan.period == "1y"
        assert collector.yf_params == {"period": "1y"}

    def test_yfinance_collector_initialization_with_start_end(self):
        """Test YFinanceCollector initialization with start and end dates."""
        collector = YFinanceCollector(
            symbol="AAPL", start="2023-01-01", end="2023-12-31"
        )

        assert collector.collection_plan.symbol == "AAPL"
        assert collector.collection_plan.start == "2023-01-01"
        assert collector.collection_plan.end == "2023-12-31"
        assert collector.yf_params == {"start": "2023-01-01", "end": "2023-12-31"}

    def test_financial_data_collector_protocol_compliance(self):
        """Test that YFinanceCollector implements the FinancialDataCollector protocol."""
        collection_plan = YFinanceCollectionPlan(symbol="AAPL", period="1y")
        collector = YFinanceCollector(collection_plan=collection_plan)

        # Verify that the collector has the required method
        assert hasattr(collector, "get_historical_data")
        assert callable(collector.get_historical_data)

        # Verify the method signature (basic check)
        sig = inspect.signature(
            YFinanceCollector.get_historical_data
        )  # Use class method, not instance method
        assert len(sig.parameters) == 1  # Only self parameter
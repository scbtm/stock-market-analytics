import pytest
from pydantic import ValidationError

from stock_market_analytics.data_collection.models import YFinanceCollectionPlan


class TestYFinanceCollectionPlan:
    """Test suite for YFinanceCollectionPlan model validation and functionality."""

    def test_valid_period_only(self):
        """Test valid collection plan with period only. This shouldn't crash."""
        plan = YFinanceCollectionPlan(symbol="AAPL", period="1y")
        assert plan.period == "1y"
        assert plan.start is None
        assert plan.end is None
        assert plan.interval is None
        assert plan.to_yfinance_params() == {"period": "1y"}

    def test_valid_start_only(self):
        """Test valid collection plan with start only. This shouldn't crash."""
        plan = YFinanceCollectionPlan(symbol="AAPL", start="2023-01-01")
        assert plan.start == "2023-01-01"
        assert plan.period is None
        assert plan.end is None
        assert plan.interval is None
        assert plan.to_yfinance_params() == {"start": "2023-01-01"}

    def test_valid_start_and_ends(self):
        """Test valid collection plan with both start and end dates. This shouldn't crash."""
        plan = YFinanceCollectionPlan(
            symbol="AAPL", start="2023-01-01", end="2023-12-31"
        )
        assert plan.start == "2023-01-01"
        assert plan.end == "2023-12-31"
        assert plan.period is None
        assert plan.interval is None
        assert plan.to_yfinance_params() == {"start": "2023-01-01", "end": "2023-12-31"}

    def test_valid_period_with_interval(self):
        """Test valid collection plan with period and interval. This shouldn't crash."""
        plan = YFinanceCollectionPlan(symbol="AAPL", period="1mo", interval="1d")
        assert plan.period == "1mo"
        assert plan.interval == "1d"
        assert plan.start is None
        assert plan.end is None
        assert plan.to_yfinance_params() == {"period": "1mo", "interval": "1d"}

    def test_valid_start_with_interval(self):
        """Test valid collection plan with start and interval. This shouldn't crash."""
        plan = YFinanceCollectionPlan(symbol="AAPL", start="2023-01-01", interval="1h")
        assert plan.start == "2023-01-01"
        assert plan.interval == "1h"
        assert plan.period is None
        assert plan.end is None
        assert plan.to_yfinance_params() == {"start": "2023-01-01", "interval": "1h"}

    def test_invalid_date_format_start(self):
        """Test that invalid start format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            YFinanceCollectionPlan(symbol="AAPL", start="24-2023-01")
        assert "Date must be in YYYY-MM-DD format" in str(exc_info.value)

    def test_invalid_date_format_end(self):
        """Test that invalid end format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            YFinanceCollectionPlan(symbol="AAPL", end="2023/01/01")

        assert "Date must be in YYYY-MM-DD format" in str(exc_info.value)

    def test_no_parameters_provided(self):
        """Test that providing no parameters raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            YFinanceCollectionPlan(symbol="AAPL")

        assert "At least one parameter" in str(exc_info.value)

    def test_missing_start_and_period(self):
        """Test that missing both start and period raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            YFinanceCollectionPlan(symbol="AAPL", end="2023-12-31")

        assert "Either start or period must be provided" in str(exc_info.value)

    def test_both_start_and_period_provided(self):
        """Test that providing both start and period raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            YFinanceCollectionPlan(symbol="AAPL", start="2023-01-01", period="1y")

        assert "start and period cannot both be provided" in str(exc_info.value)

    def test_both_end_and_period_provided(self):
        """Test that providing both end and period raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            YFinanceCollectionPlan(symbol="AAPL", end="2023-12-31", period="1y")

        assert "end and period cannot both be provided" in str(exc_info.value)

    def test_start_after_end(self):
        """Test that start after end raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            YFinanceCollectionPlan(symbol="AAPL", start="2023-12-31", end="2023-01-01")

        assert "start must be before end" in str(exc_info.value)

    def test_start_equals_end(self):
        """Test that start equal to end raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            YFinanceCollectionPlan(symbol="AAPL", start="2023-01-01", end="2023-01-01")

        assert "start must be before end" in str(exc_info.value)

    def test_interval_with_start_and_ends(self):
        """Test that providing interval with both start and end dates raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            YFinanceCollectionPlan(
                symbol="AAPL", start="2023-01-01", end="2023-12-31", interval="1d"
            )

        assert "interval should not be provided when using both start and end" in str(
            exc_info.value
        )

    def test_valid_intervals(self):
        """Test all valid interval values."""
        valid_intervals = [
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ]

        for interval in valid_intervals:
            plan = YFinanceCollectionPlan(symbol="AAPL", period="1y", interval=interval)
            assert plan.interval == interval

    def test_valid_periods(self):
        """Test all valid period values."""
        valid_periods = [
            "1d",
            "5d",
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y",
            "10y",
            "ytd",
            "max",
        ]

        for period in valid_periods:
            plan = YFinanceCollectionPlan(symbol="AAPL", period=period)
            assert plan.period == period

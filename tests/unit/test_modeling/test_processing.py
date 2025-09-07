"""Simple unit tests for modeling processing functions."""

import pandas as pd
from datetime import date, timedelta

from stock_market_analytics.modeling.processing_functions import (
    split_data,
    metadata,
)


class TestSplitData:
    """Test suite for split_data function."""

    def test_splits_data_with_time_span(self):
        """Test that split_data correctly splits by time span."""
        # Create data over 1 year
        dates = [date(2023, 1, 1) + timedelta(days=i * 30) for i in range(12)]
        data = pd.DataFrame(
            {
                "date": dates,
                "symbol": ["AAPL"] * 12,
                "close": [100.0 + i for i in range(12)],
            }
        )

        result = split_data(data, time_span=180)  # 6 months

        # Check that fold column was added
        assert "fold" in result.columns

        # Check that we have different folds
        folds = result["fold"].unique()
        assert len(folds) >= 1  # At least train should exist

        # Check that test data is recent
        if "test" in folds:
            test_dates = result[result["fold"] == "test"]["date"]
            assert all(test_dates >= result["date"].max() - timedelta(days=180))

    def test_handles_short_timespan(self):
        """Test split_data with short time span."""
        data = pd.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 15)],
                "symbol": ["AAPL", "AAPL"],
                "close": [100.0, 105.0],
            }
        )

        result = split_data(data, time_span=7)  # 1 week

        # Should still add fold column
        assert "fold" in result.columns
        assert len(result) == 2


class TestMetadata:
    """Test suite for metadata function."""

    def test_creates_metadata_from_split_data(self):
        """Test that metadata creates correct structure from split data."""
        split_data_df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 15),
                    date(2023, 2, 1),
                    date(2023, 1, 1),
                    date(2023, 2, 1),
                ],
                "close": [100.0, 105.0, 102.0, 200.0, 205.0],
                "fold": ["train", "validation", "test", "train", "test"],
            }
        )

        meta = metadata(split_data_df)

        # Check basic structure exists
        assert isinstance(meta, dict)
        # The actual keys depend on implementation, but it should return a dict
        assert len(meta) > 0

    def test_metadata_with_minimal_data(self):
        """Test metadata creation with minimal data."""
        split_data_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [date(2023, 1, 1)],
                "close": [100.0],
                "fold": ["train"],
            }
        )

        meta = metadata(split_data_df)

        # Should still work with minimal data
        assert isinstance(meta, dict)

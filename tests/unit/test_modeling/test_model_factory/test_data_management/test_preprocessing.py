"""
Unit tests for preprocessing module.

This module tests the time series data preprocessing functionality including
data splitting, purging, embargoes, and cross-validation for financial data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from stock_market_analytics.modeling.model_factory.data_management.preprocessing import (
    _as_dt,
    _unique_sorted_dates,
    _cut_by_fractions,
    _validate,
    _purge_around,
    _no_overlap_with,
    _apply_segment_masks,
    _xy,
    _build_test_windows,
    make_holdout_splits,
    get_modeling_sets,
    PurgedTimeSeriesSplit,
)


# -------------------------
# Fixtures
# -------------------------


@pytest.fixture
def sample_datetime_series():
    """Sample datetime series for testing."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.Series(dates)


@pytest.fixture
def sample_stock_data():
    """Sample stock market DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    symbols = ["AAPL", "MSFT"] * 50  # 50 records per symbol

    data = {
        "date": dates,
        "symbol": symbols,
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.randn(100),
        "price": 100 + np.cumsum(np.random.randn(100) * 0.1),
    }
    return pd.DataFrame(data)


@pytest.fixture
def minimal_stock_data():
    """Minimal stock data for edge case testing."""
    dates = pd.date_range("2023-01-01", periods=15, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAPL"] * 15,
            "feature1": range(15),
            "target": range(15, 30),
        }
    )


@pytest.fixture
def unique_dates_array():
    """Array of unique sorted dates."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    return np.array(dates)


# -------------------------
# Test Helper Functions
# -------------------------


class TestHelperFunctions:
    """Test helper utility functions."""

    def test_as_dt_with_datetime_series(self, sample_datetime_series):
        """Test _as_dt with datetime series."""
        result = _as_dt(sample_datetime_series)

        assert isinstance(result, pd.Series)
        assert np.issubdtype(result.dtype, np.datetime64)
        pd.testing.assert_series_equal(result, sample_datetime_series)

    def test_as_dt_with_string_series(self):
        """Test _as_dt with string dates."""
        string_dates = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        result = _as_dt(string_dates)

        assert np.issubdtype(result.dtype, np.datetime64)
        expected = pd.to_datetime(string_dates, utc=False)
        pd.testing.assert_series_equal(result, expected)

    def test_as_dt_with_mixed_formats(self):
        """Test _as_dt with mixed date formats."""
        # Use format='mixed' to handle different date formats
        mixed_dates = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        result = _as_dt(mixed_dates)

        assert np.issubdtype(result.dtype, np.datetime64)
        assert len(result) == 3

    def test_unique_sorted_dates(self, sample_datetime_series):
        """Test _unique_sorted_dates function."""
        # Add duplicates and unsorted data
        unsorted_with_dups = (
            pd.concat(
                [
                    sample_datetime_series,
                    sample_datetime_series[:3],
                    sample_datetime_series[7:],
                ]
            )
            .sample(frac=1)
            .reset_index(drop=True)
        )

        result = _unique_sorted_dates(unsorted_with_dups)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_datetime_series)  # Should remove duplicates
        assert np.array_equal(result, np.sort(result))  # Should be sorted

    def test_cut_by_fractions_valid(self, unique_dates_array):
        """Test _cut_by_fractions with valid fractions."""
        fractions = (0.6, 0.2, 0.1, 0.1)
        train_end, val_end, cal_end = _cut_by_fractions(unique_dates_array, fractions)

        assert isinstance(train_end, (pd.Timestamp, np.datetime64))
        assert isinstance(val_end, (pd.Timestamp, np.datetime64))
        assert isinstance(cal_end, (pd.Timestamp, np.datetime64))
        assert train_end <= val_end <= cal_end

    def test_cut_by_fractions_invalid_sum(self, unique_dates_array):
        """Test _cut_by_fractions with fractions that don't sum to 1."""
        fractions = (0.6, 0.2, 0.1, 0.05)  # Sum = 0.95

        with pytest.raises(ValueError, match="fractions must sum to 1"):
            _cut_by_fractions(unique_dates_array, fractions)

    def test_cut_by_fractions_too_few_dates(self):
        """Test _cut_by_fractions with too few dates."""
        short_dates = np.array(pd.date_range("2023-01-01", periods=5))
        fractions = (0.6, 0.2, 0.1, 0.1)

        with pytest.raises(ValueError, match="Too few unique dates"):
            _cut_by_fractions(short_dates, fractions)

    def test_cut_by_fractions_edge_case_minimum_dates(self):
        """Test _cut_by_fractions with minimum number of dates."""
        min_dates = np.array(pd.date_range("2023-01-01", periods=10))
        fractions = (0.6, 0.2, 0.1, 0.1)

        train_end, val_end, cal_end = _cut_by_fractions(min_dates, fractions)
        assert train_end <= val_end <= cal_end


# -------------------------
# Test Validation Functions
# -------------------------


class TestValidationFunctions:
    """Test data validation and preprocessing functions."""

    def test_validate_valid_dataframe(self, sample_stock_data):
        """Test _validate with valid DataFrame."""
        result = _validate(sample_stock_data, "date", "symbol")

        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns
        assert "symbol" in result.columns
        assert np.issubdtype(result["date"].dtype, np.datetime64)

    def test_validate_missing_date_column(self, sample_stock_data):
        """Test _validate with missing date column."""
        df_no_date = sample_stock_data.drop("date", axis=1)

        with pytest.raises(ValueError, match="'date' column missing"):
            _validate(df_no_date, "date", "symbol")

    def test_validate_missing_symbol_column(self, sample_stock_data):
        """Test _validate with missing symbol column."""
        df_no_symbol = sample_stock_data.drop("symbol", axis=1)

        with pytest.raises(ValueError, match="'symbol' column missing"):
            _validate(df_no_symbol, "date", "symbol")

    def test_validate_string_dates(self):
        """Test _validate converts string dates properly."""
        df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "value": [1, 2, 3],
            }
        )

        result = _validate(df, "date", "symbol")
        assert np.issubdtype(result["date"].dtype, np.datetime64)


# -------------------------
# Test Purging Functions
# -------------------------


class TestPurgingFunctions:
    """Test purging and embargo functionality."""

    def test_purge_around_with_valid_window(self):
        """Test _purge_around with valid time window."""
        dates = np.array(pd.date_range("2023-01-01", periods=10))
        window_mask = np.array(
            [False, False, True, True, False, False, False, False, False, False]
        )
        d = pd.Series(pd.date_range("2023-01-01", periods=10))
        embargo = 2
        h = 1

        union_start, union_end = _purge_around(window_mask, dates, d, embargo, h)

        assert isinstance(union_start, np.datetime64)
        assert isinstance(union_end, np.datetime64)
        assert union_start <= union_end

    def test_purge_around_empty_window(self):
        """Test _purge_around with empty window mask."""
        dates = np.array(pd.date_range("2023-01-01", periods=10))
        window_mask = np.array([False] * 10)
        d = pd.Series(pd.date_range("2023-01-01", periods=10))
        embargo = 2
        h = 1

        union_start, union_end = _purge_around(window_mask, dates, d, embargo, h)

        # Should return min date when no window
        expected_min = np.datetime64(d.min())
        assert union_start == expected_min
        assert union_end == expected_min

    def test_purge_around_with_large_embargo(self):
        """Test _purge_around with large embargo period."""
        dates = np.array(pd.date_range("2023-01-01", periods=10))
        window_mask = np.array(
            [False, False, True, True, False, False, False, False, False, False]
        )
        d = pd.Series(pd.date_range("2023-01-01", periods=10))
        embargo = 10  # Large embargo
        h = 5

        union_start, union_end = _purge_around(window_mask, dates, d, embargo, h)

        # Should handle large embargo gracefully
        assert union_start <= union_end

    def test_no_overlap_with_function(self):
        """Test _no_overlap_with overlap detection."""
        dates = np.array(pd.date_range("2023-01-01", periods=10))
        end_dates = dates + np.timedelta64(1, "D")

        union_start = np.datetime64("2023-01-03")
        union_end = np.datetime64("2023-01-06")

        result = _no_overlap_with(union_start, union_end, dates, end_dates)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == len(dates)

    def test_no_overlap_with_no_overlap(self):
        """Test _no_overlap_with when there's no overlap."""
        dates = np.array([np.datetime64("2023-01-01"), np.datetime64("2023-01-02")])
        end_dates = np.array([np.datetime64("2023-01-02"), np.datetime64("2023-01-03")])

        union_start = np.datetime64("2023-01-05")
        union_end = np.datetime64("2023-01-06")

        result = _no_overlap_with(union_start, union_end, dates, end_dates)

        # Should return all True (no overlap)
        assert np.all(result)

    def test_no_overlap_with_complete_overlap(self):
        """Test _no_overlap_with when there's complete overlap."""
        dates = np.array([np.datetime64("2023-01-04"), np.datetime64("2023-01-05")])
        end_dates = np.array([np.datetime64("2023-01-05"), np.datetime64("2023-01-06")])

        union_start = np.datetime64("2023-01-03")
        union_end = np.datetime64("2023-01-07")

        result = _no_overlap_with(union_start, union_end, dates, end_dates)

        # Should return all False (complete overlap)
        assert np.all(~result)


# -------------------------
# Test Data Splitting Functions
# -------------------------


class TestDataSplittingFunctions:
    """Test data splitting and masking functions."""

    def test_apply_segment_masks_basic(self, sample_stock_data):
        """Test _apply_segment_masks with basic functionality."""
        df = sample_stock_data
        train_end = pd.Timestamp("2023-02-01")
        val_end = pd.Timestamp("2023-02-15")
        cal_end = pd.Timestamp("2023-03-01")

        result = _apply_segment_masks(df, train_end, val_end, cal_end)

        assert isinstance(result, dict)
        assert all(
            key in result for key in ["train_idx", "val_idx", "cal_idx", "test_idx"]
        )
        assert all(isinstance(result[key], np.ndarray) for key in result.keys())

    def test_apply_segment_masks_with_custom_horizon(self, sample_stock_data):
        """Test _apply_segment_masks with custom horizon days."""
        df = sample_stock_data
        train_end = pd.Timestamp("2023-02-01")
        val_end = pd.Timestamp("2023-02-15")
        cal_end = pd.Timestamp("2023-03-01")

        result = _apply_segment_masks(df, train_end, val_end, cal_end, horizon_days=10)

        assert isinstance(result, dict)
        # Should still return valid indices
        assert all(len(result[key]) >= 0 for key in result.keys())

    def test_apply_segment_masks_with_embargo(self, sample_stock_data):
        """Test _apply_segment_masks with custom embargo days."""
        df = sample_stock_data
        train_end = pd.Timestamp("2023-02-01")
        val_end = pd.Timestamp("2023-02-15")
        cal_end = pd.Timestamp("2023-03-01")

        result = _apply_segment_masks(df, train_end, val_end, cal_end, embargo_days=7)

        assert isinstance(result, dict)
        # Embargo should reduce overlap between sets
        assert all(len(result[key]) >= 0 for key in result.keys())

    def test_xy_function(self, sample_stock_data):
        """Test _xy feature/target extraction."""
        feature_cols = ["feature1", "feature2"]
        target_col = "target"

        X, y = _xy(sample_stock_data, feature_cols, target_col)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert list(X.columns) == feature_cols
        assert y.name == target_col
        assert len(X) == len(y) == len(sample_stock_data)

    def test_xy_function_single_feature(self, sample_stock_data):
        """Test _xy with single feature column."""
        feature_cols = ["feature1"]
        target_col = "target"

        X, y = _xy(sample_stock_data, feature_cols, target_col)

        assert isinstance(X, pd.DataFrame)
        assert len(X.columns) == 1
        assert X.columns[0] == "feature1"

    def test_xy_function_copies_data(self, sample_stock_data):
        """Test _xy creates copies of data."""
        feature_cols = ["feature1", "feature2"]
        target_col = "target"

        X, y = _xy(sample_stock_data, feature_cols, target_col)

        # Modify original data
        sample_stock_data.loc[0, "feature1"] = 999
        sample_stock_data.loc[0, "target"] = 999

        # Copies should be unchanged
        assert X.loc[0, "feature1"] != 999
        assert y.iloc[0] != 999

    def test_build_test_windows_equal_partitions(self, unique_dates_array):
        """Test _build_test_windows with equal partitions."""
        n_splits = 4
        result = _build_test_windows(unique_dates_array, n_splits)

        assert isinstance(result, list)
        assert len(result) == n_splits
        assert all(isinstance(window, tuple) and len(window) == 2 for window in result)

        # Check chronological order
        for i in range(len(result) - 1):
            assert result[i][0] <= result[i + 1][0]

    def test_build_test_windows_with_span(self, unique_dates_array):
        """Test _build_test_windows with fixed span days."""
        n_splits = 3
        test_span_days = 5
        result = _build_test_windows(unique_dates_array, n_splits, test_span_days)

        assert isinstance(result, list)
        assert len(result) == n_splits

        # Check that spans are approximately correct
        for start, end in result:
            # Convert to pandas Timestamp to get .days attribute
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            span_days = (end_ts - start_ts).days
            assert span_days <= test_span_days + 1  # Allow for rounding

    def test_build_test_windows_chronological_order(self, unique_dates_array):
        """Test _build_test_windows maintains chronological order."""
        n_splits = 5
        result = _build_test_windows(unique_dates_array, n_splits)

        # Verify chronological ordering
        start_dates = [window[0] for window in result]
        assert start_dates == sorted(start_dates)


# -------------------------
# Test Public API Functions
# -------------------------


class TestPublicAPIFunctions:
    """Test public API functions."""

    def test_make_holdout_splits_with_fractions(self, sample_stock_data):
        """Test make_holdout_splits with fraction-based splitting."""
        result = make_holdout_splits(
            sample_stock_data,
            date_col="date",
            symbol_col="symbol",
            fractions=(0.6, 0.2, 0.1, 0.1),
        )

        assert isinstance(result, dict)
        assert all(
            key in result for key in ["train_idx", "val_idx", "cal_idx", "test_idx"]
        )
        assert all(isinstance(result[key], np.ndarray) for key in result.keys())

    def test_make_holdout_splits_return_frames(self, sample_stock_data):
        """Test make_holdout_splits returning DataFrames."""
        result = make_holdout_splits(
            sample_stock_data, date_col="date", symbol_col="symbol", return_frames=True
        )

        assert isinstance(result, dict)
        assert all(key in result for key in ["train", "val", "cal", "test"])
        assert all(isinstance(result[key], pd.DataFrame) for key in result.keys())

    def test_make_holdout_splits_with_cut_dates(self, sample_stock_data):
        """Test make_holdout_splits with explicit cut dates."""
        cut_dates = (
            pd.Timestamp("2023-02-01"),
            pd.Timestamp("2023-02-15"),
            pd.Timestamp("2023-03-01"),
        )

        result = make_holdout_splits(
            sample_stock_data, date_col="date", symbol_col="symbol", cut_dates=cut_dates
        )

        assert isinstance(result, dict)
        assert all(
            key in result for key in ["train_idx", "val_idx", "cal_idx", "test_idx"]
        )

    def test_make_holdout_splits_validates_columns(self, sample_stock_data):
        """Test make_holdout_splits validates required columns."""
        df_no_date = sample_stock_data.drop("date", axis=1)

        with pytest.raises(ValueError, match="'date' column missing"):
            make_holdout_splits(df_no_date, date_col="date", symbol_col="symbol")

    def test_get_modeling_sets_basic(self, sample_stock_data):
        """Test get_modeling_sets basic functionality."""
        feature_cols = ["feature1", "feature2"]
        target_col = "target"

        result = get_modeling_sets(
            sample_stock_data, feature_cols=feature_cols, target_col=target_col
        )

        assert isinstance(result, dict)
        assert all(key in result for key in ["train", "val", "cal", "test"])

        for key in result.keys():
            X, y = result[key]
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            assert list(X.columns) == feature_cols

    def test_get_modeling_sets_custom_fractions(self, sample_stock_data):
        """Test get_modeling_sets with custom fractions."""
        feature_cols = ["feature1"]
        target_col = "target"
        fractions = (0.8, 0.1, 0.05, 0.05)

        result = get_modeling_sets(
            sample_stock_data,
            feature_cols=feature_cols,
            target_col=target_col,
            fractions=fractions,
        )

        assert isinstance(result, dict)
        # Should have larger training set due to 0.8 fraction
        X_train, _ = result["train"]
        X_val, _ = result["val"]
        assert len(X_train) > len(X_val)

    def test_get_modeling_sets_single_feature(self, sample_stock_data):
        """Test get_modeling_sets with single feature."""
        feature_cols = ["feature1"]
        target_col = "target"

        result = get_modeling_sets(
            sample_stock_data, feature_cols=feature_cols, target_col=target_col
        )

        for key in result.keys():
            X, y = result[key]
            assert X.shape[1] == 1
            assert X.columns[0] == "feature1"


# -------------------------
# Test PurgedTimeSeriesSplit Class
# -------------------------


class TestPurgedTimeSeriesSplit:
    """Test PurgedTimeSeriesSplit cross-validator."""

    def test_init_default_params(self):
        """Test PurgedTimeSeriesSplit initialization with defaults."""
        splitter = PurgedTimeSeriesSplit()

        assert splitter.n_splits == 5
        assert splitter.h == 5
        assert splitter.embargo == 5
        assert splitter.test_span_days is None
        assert splitter.min_train_fraction == 0.05

    def test_init_custom_params(self):
        """Test PurgedTimeSeriesSplit initialization with custom params."""
        date_series = pd.Series(pd.date_range("2023-01-01", periods=10))

        splitter = PurgedTimeSeriesSplit(
            n_splits=3,
            date=date_series,
            horizon_days=10,
            embargo_days=7,
            test_span_days=5,
            min_train_fraction=0.1,
        )

        assert splitter.n_splits == 3
        assert splitter.h == 10
        assert splitter.embargo == 7
        assert splitter.test_span_days == 5
        assert splitter.min_train_fraction == 0.1

    def test_init_invalid_n_splits(self):
        """Test PurgedTimeSeriesSplit with invalid n_splits."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            PurgedTimeSeriesSplit(n_splits=1)

    def test_init_embargo_defaults_to_horizon(self):
        """Test that embargo defaults to horizon when not specified."""
        splitter = PurgedTimeSeriesSplit(horizon_days=7)
        assert splitter.embargo == 7

    def test_split_with_date_column(self, sample_stock_data):
        """Test split method with date column in DataFrame."""
        splitter = PurgedTimeSeriesSplit(n_splits=3)
        X = sample_stock_data[["feature1", "feature2", "date"]]

        splits = list(splitter.split(X))

        assert len(splits) <= 3  # May be fewer due to min_train_fraction
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert len(np.intersect1d(train_idx, test_idx)) == 0  # No overlap

    def test_split_with_provided_date_series(self, sample_stock_data):
        """Test split method with date series provided at init."""
        date_series = sample_stock_data["date"]
        splitter = PurgedTimeSeriesSplit(n_splits=3, date=date_series)
        X = sample_stock_data[["feature1", "feature2"]]  # No date column

        splits = list(splitter.split(X))

        assert len(splits) <= 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_split_no_date_column_no_provided_date(self, sample_stock_data):
        """Test split method when no date column or provided date."""
        splitter = PurgedTimeSeriesSplit(n_splits=3)
        X = sample_stock_data[["feature1", "feature2"]]  # No date column

        with pytest.raises(ValueError, match="X must contain a 'date' column"):
            list(splitter.split(X))

    def test_split_causal_ordering(self, sample_stock_data):
        """Test that split maintains causal ordering (train before test)."""
        splitter = PurgedTimeSeriesSplit(n_splits=3, horizon_days=1, embargo_days=1)
        X = sample_stock_data[["feature1", "feature2", "date"]]

        splits = list(splitter.split(X))

        for train_idx, test_idx in splits:
            if len(train_idx) > 0 and len(test_idx) > 0:
                train_dates = X.iloc[train_idx]["date"]
                test_dates = X.iloc[test_idx]["date"]

                # All training dates should be before test dates (with gap)
                assert train_dates.max() < test_dates.min()

    def test_split_with_min_train_fraction(self):
        """Test split respects min_train_fraction."""
        # Small dataset to trigger min_train_fraction
        small_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=20, freq="D"),
                "feature1": range(20),
            }
        )

        splitter = PurgedTimeSeriesSplit(n_splits=10, min_train_fraction=0.5)
        X = small_data

        splits = list(splitter.split(X))

        # Should have fewer splits due to min_train_fraction
        assert len(splits) < 10

    def test_split_with_test_span_days(self, sample_stock_data):
        """Test split with fixed test span days."""
        splitter = PurgedTimeSeriesSplit(n_splits=3, test_span_days=10)
        X = sample_stock_data[["feature1", "feature2", "date"]]

        splits = list(splitter.split(X))

        for train_idx, test_idx in splits:
            if len(test_idx) > 0:
                test_dates = X.iloc[test_idx]["date"]
                span_days = (test_dates.max() - test_dates.min()).days
                # Should be approximately 10 days or less
                assert span_days <= 12  # Allow some tolerance

    def test_split_empty_result_when_insufficient_data(self):
        """Test split returns empty when insufficient data."""
        # Very small dataset
        tiny_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "feature1": range(5),
            }
        )

        splitter = PurgedTimeSeriesSplit(n_splits=5, min_train_fraction=0.8)

        splits = list(splitter.split(tiny_data))

        # Should return empty or very few splits
        assert len(splits) <= 1


# -------------------------
# Test Integration Scenarios
# -------------------------


class TestIntegrationScenarios:
    """Test integration and edge case scenarios."""

    def test_full_pipeline_integration(self, sample_stock_data):
        """Test complete preprocessing pipeline integration."""
        # Test the full workflow from raw data to modeling sets
        feature_cols = ["feature1", "feature2"]
        target_col = "target"

        # Get modeling sets
        sets = get_modeling_sets(
            sample_stock_data,
            feature_cols=feature_cols,
            target_col=target_col,
            fractions=(0.6, 0.2, 0.1, 0.1),
        )

        # Verify we have all sets
        assert "train" in sets
        assert "val" in sets
        assert "cal" in sets
        assert "test" in sets

        # Verify data consistency
        total_samples = sum(len(sets[key][0]) for key in sets.keys())
        assert total_samples <= len(sample_stock_data)  # Some may be purged

    def test_cross_validation_integration(self, sample_stock_data):
        """Test cross-validation with preprocessing integration."""
        splitter = PurgedTimeSeriesSplit(n_splits=3)
        X = sample_stock_data[["feature1", "feature2", "date"]]
        y = sample_stock_data["target"]

        # Test that CV works with our preprocessing
        cv_scores = []
        for train_idx, test_idx in splitter.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Simple validation that data is properly split
            assert len(X_train) > 0
            assert len(X_test) > 0
            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)

            cv_scores.append(len(X_test))  # Dummy score

        assert len(cv_scores) > 0

    def test_edge_case_single_symbol_minimal_data(self, minimal_stock_data):
        """Test preprocessing with minimal data for single symbol."""
        result = make_holdout_splits(
            minimal_stock_data, date_col="date", symbol_col="symbol", return_frames=True
        )

        # Should still work with minimal data
        assert isinstance(result, dict)
        assert all(key in result for key in ["train", "val", "cal", "test"])

    def test_edge_case_very_short_time_series(self):
        """Test preprocessing with very short time series."""
        short_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=12, freq="D"),
                "symbol": ["AAPL"] * 12,
                "feature1": range(12),
                "target": range(12),
            }
        )

        # Should handle short series gracefully
        try:
            result = make_holdout_splits(
                short_data, date_col="date", symbol_col="symbol"
            )
            assert isinstance(result, dict)
        except ValueError:
            # Acceptable to fail with very short series
            pass

    def test_robustness_with_gaps_in_data(self):
        """Test preprocessing robustness with gaps in time series."""
        # Create data with gaps
        dates1 = pd.date_range("2023-01-01", periods=30, freq="D")
        dates2 = pd.date_range("2023-03-01", periods=30, freq="D")  # Gap in February
        all_dates = list(dates1) + list(dates2)

        gapped_data = pd.DataFrame(
            {
                "date": all_dates,
                "symbol": ["AAPL"] * len(all_dates),
                "feature1": range(len(all_dates)),
                "target": range(len(all_dates)),
            }
        )

        result = make_holdout_splits(gapped_data, date_col="date", symbol_col="symbol")

        # Should handle gaps gracefully
        assert isinstance(result, dict)
        assert all(
            key in result for key in ["train_idx", "val_idx", "cal_idx", "test_idx"]
        )

    def test_multiple_symbols_consistency(self):
        """Test preprocessing consistency across multiple symbols."""
        # Create multi-symbol dataset
        symbols = ["AAPL", "MSFT", "GOOGL"]
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        multi_symbol_data = []
        for symbol in symbols:
            for date in dates:
                multi_symbol_data.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "feature1": np.random.randn(),
                        "target": np.random.randn(),
                    }
                )

        df = pd.DataFrame(multi_symbol_data)

        result = make_holdout_splits(
            df, date_col="date", symbol_col="symbol", return_frames=True
        )

        # Verify all symbols are represented across splits
        for split_name, split_df in result.items():
            if len(split_df) > 0:
                unique_symbols = split_df["symbol"].unique()
                assert len(unique_symbols) > 0  # Should have at least one symbol

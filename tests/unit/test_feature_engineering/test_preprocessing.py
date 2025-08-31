from datetime import date

import polars as pl
import pytest

from stock_market_analytics.feature_engineering.preprocessing import (
    base,
    interpolated,
    dfp,
    log_returns_d,
    dollar_volume,
    y_log_returns,
    dff
)


class TestBaseFunction:
    """Test suite for the base preprocessing function."""

    @pytest.fixture
    def sample_unsorted_data(self):
        """Create sample unsorted data for testing."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL", "AAPL", "GOOGL", "AAPL"],
                "date": [
                    date(2023, 1, 3),
                    date(2023, 1, 2),
                    date(2023, 1, 1),
                    date(2023, 1, 3),
                    date(2023, 1, 2),
                ],
                "open": [152.0, 101.0, 150.0, 103.0, 151.0],
                "close": [154.0, 102.0, 153.0, 104.0, 152.0],
                "volume": [1200000, 2100000, 1000000, 2300000, 1100000],
            }
        )

    @pytest.fixture
    def sample_already_sorted_data(self):
        """Create sample data that's already sorted."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                ],
                "open": [150.0, 151.0, 152.0, 101.0, 103.0],
                "close": [153.0, 152.0, 154.0, 102.0, 104.0],
                "volume": [1000000, 1100000, 1200000, 2100000, 2300000],
            }
        )

    @pytest.fixture
    def empty_dataframe(self):
        """Create an empty DataFrame."""
        return pl.DataFrame(
            {
                "symbol": [],
                "date": [],
                "open": [],
                "close": [],
                "volume": [],
            }
        )

    @pytest.fixture
    def single_row_data(self):
        """Create single row DataFrame."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [date(2023, 1, 1)],
                "open": [150.0],
                "close": [153.0],
                "volume": [1000000],
            }
        )

    @pytest.fixture
    def single_symbol_data(self):
        """Create data with only one symbol."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "date": [date(2023, 1, 3), date(2023, 1, 1), date(2023, 1, 2)],
                "open": [152.0, 150.0, 151.0],
                "close": [154.0, 153.0, 152.0],
                "volume": [1200000, 1000000, 1100000],
            }
        )

    @pytest.fixture
    def duplicate_symbol_date_data(self):
        """Create data with duplicate symbol-date combinations."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                ],
                "open": [150.0, 151.0, 150.5, 151.5],
                "close": [153.0, 152.0, 153.5, 152.5],
                "volume": [1000000, 1100000, 1050000, 1150000],
            }
        )

    def test_base_sorts_unsorted_data_correctly(self, sample_unsorted_data):
        """Test that base function sorts data by symbol and date."""
        result = base(sample_unsorted_data)

        expected_order = [
            ("AAPL", date(2023, 1, 1)),
            ("AAPL", date(2023, 1, 2)),
            ("AAPL", date(2023, 1, 3)),
            ("GOOGL", date(2023, 1, 2)),
            ("GOOGL", date(2023, 1, 3)),
        ]

        actual_order = list(zip(result["symbol"].to_list(), result["date"].to_list(), strict=False))
        assert actual_order == expected_order

    def test_base_maintains_already_sorted_data(self, sample_already_sorted_data):
        """Test that base function maintains order of already sorted data."""
        result = base(sample_already_sorted_data)

        expected_order = [
            ("AAPL", date(2023, 1, 1)),
            ("AAPL", date(2023, 1, 2)),
            ("AAPL", date(2023, 1, 3)),
            ("GOOGL", date(2023, 1, 2)),
            ("GOOGL", date(2023, 1, 3)),
        ]

        actual_order = list(zip(result["symbol"].to_list(), result["date"].to_list(), strict=False))
        assert actual_order == expected_order

    def test_base_handles_empty_dataframe(self, empty_dataframe):
        """Test that base function handles empty DataFrame correctly."""
        result = base(empty_dataframe)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert result.columns == empty_dataframe.columns

    def test_base_handles_single_row(self, single_row_data):
        """Test that base function handles single row DataFrame."""
        result = base(single_row_data)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        assert result.equals(single_row_data)

    def test_base_sorts_single_symbol_multiple_dates(self, single_symbol_data):
        """Test sorting with single symbol but multiple dates."""
        result = base(single_symbol_data)

        expected_dates = [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)]
        actual_dates = result["date"].to_list()

        assert actual_dates == expected_dates
        assert all(symbol == "AAPL" for symbol in result["symbol"].to_list())

    def test_base_preserves_all_columns(self, sample_unsorted_data):
        """Test that base function preserves all columns from input."""
        result = base(sample_unsorted_data)

        assert result.columns == sample_unsorted_data.columns
        assert len(result) == len(sample_unsorted_data)

    def test_base_preserves_data_integrity(self, sample_unsorted_data):
        """Test that base function doesn't modify the actual data values."""
        result = base(sample_unsorted_data)

        original_symbols = set(sample_unsorted_data["symbol"].to_list())
        result_symbols = set(result["symbol"].to_list())
        assert original_symbols == result_symbols

        original_dates = set(sample_unsorted_data["date"].to_list())
        result_dates = set(result["date"].to_list())
        assert original_dates == result_dates

    def test_base_handles_duplicate_symbol_date_combinations(
        self, duplicate_symbol_date_data
    ):
        """Test that base function handles duplicate symbol-date combinations."""
        result = base(duplicate_symbol_date_data)

        assert len(result) == len(duplicate_symbol_date_data)
        sorted_pairs = list(zip(result["symbol"].to_list(), result["date"].to_list(), strict=False))

        for i in range(len(sorted_pairs) - 1):
            current_symbol, current_date = sorted_pairs[i]
            next_symbol, next_date = sorted_pairs[i + 1]

            if current_symbol == next_symbol:
                assert current_date <= next_date
            else:
                assert current_symbol <= next_symbol

    def test_base_returns_new_dataframe(self, sample_unsorted_data):
        """Test that base function returns a new DataFrame object."""
        result = base(sample_unsorted_data)

        assert result is not sample_unsorted_data
        assert isinstance(result, pl.DataFrame)

    def test_base_with_various_data_types(self):
        """Test base function with various column data types."""
        mixed_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL", "AAPL"],
                "date": [date(2023, 1, 3), date(2023, 1, 1), date(2023, 1, 2)],
                "open": [152.0, 101.5, 151.25],
                "close": [154.99, 102.01, 152.75],
                "volume": [1200000, 2100000, 1100000],
                "adj_close": [154.5, 101.8, 152.3],
            }
        )

        result = base(mixed_data)

        expected_order = [
            ("AAPL", date(2023, 1, 2)),
            ("AAPL", date(2023, 1, 3)),
            ("GOOGL", date(2023, 1, 1)),
        ]

        actual_order = list(zip(result["symbol"].to_list(), result["date"].to_list(), strict=False))
        assert actual_order == expected_order


class TestInterpolatedFunction:
    """Test suite for the interpolated (interpolate) preprocessing function."""

    @pytest.fixture
    def data_with_nulls(self):
        """Create sample data with null values for interpolation testing."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 4),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                ],
                "open": [150.0, None, None, 153.0, 100.0, None, 102.0],
                "close": [154.0, None, 156.0, None, 101.0, 101.5, None],
                "volume": [1000000, None, 1200000, 1300000, 2000000, 2100000, None],
            }
        )

    @pytest.fixture
    def data_without_nulls(self):
        """Create sample data without null values."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                ],
                "open": [150.0, 151.0, 152.0, 100.0, 101.0],
                "close": [154.0, 155.0, 156.0, 101.0, 102.0],
                "volume": [1000000, 1100000, 1200000, 2000000, 2100000],
            }
        )

    @pytest.fixture
    def data_all_nulls_per_symbol(self):
        """Create data where entire columns are null for specific symbols."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "GOOGL", "GOOGL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                ],
                "open": [None, None, 100.0, 101.0],
                "close": [154.0, 155.0, None, None],
                "volume": [1000000, 1100000, 2000000, 2100000],
            }
        )

    @pytest.fixture
    def empty_dataframe(self):
        """Create an empty DataFrame."""
        return pl.DataFrame(
            {
                "symbol": [],
                "date": [],
                "open": [],
                "close": [],
                "volume": [],
            }
        )

    @pytest.fixture
    def single_row_data(self):
        """Create single row DataFrame with nulls."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [date(2023, 1, 1)],
                "open": [None],
                "close": [153.0],
                "volume": [1000000],
            }
        )

    @pytest.fixture
    def data_first_last_nulls(self):
        """Create data with nulls at the beginning and end of sequences."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 4),
                    date(2023, 1, 5),
                ],
                "open": [None, 151.0, 152.0, 153.0, None],
                "close": [None, 155.0, 156.0, 157.0, None],
                "volume": [None, 1100000, 1200000, 1300000, None],
            }
        )

    def test_interpolated_interpolates_nulls_correctly(self, data_with_nulls):
        """Test that interpolated function interpolates null values correctly within each symbol group."""
        result = interpolated(data_with_nulls)

        aapl_data = result.filter(pl.col("symbol") == "AAPL")
        googl_data = result.filter(pl.col("symbol") == "GOOGL")

        # AAPL open: 150.0, None, None, 153.0 -> 150.0, 151.0, 152.0, 153.0
        assert aapl_data["open"][1] == 151.0
        assert aapl_data["open"][2] == 152.0

        # GOOGL open: 100.0, None, 102.0 -> 100.0, 101.0, 102.0
        assert googl_data["open"][1] == 101.0

        # AAPL close: 154.0, None, 156.0, None -> 154.0, 155.0, 156.0, None (boundary null stays)
        assert aapl_data["close"][1] == 155.0
        assert aapl_data["close"][3] is None  # Boundary null should remain None

    def test_interpolated_preserves_non_null_values(self, data_with_nulls):
        """Test that interpolated function preserves existing non-null values."""
        result = interpolated(data_with_nulls)

        original_non_nulls = data_with_nulls.drop_nulls()
        result_subset = result.select(original_non_nulls.columns)

        for row_idx in range(len(data_with_nulls)):
            for col in data_with_nulls.columns:
                original_val = data_with_nulls[col][row_idx]
                if original_val is not None:
                    assert result[col][row_idx] == original_val

    def test_interpolated_handles_data_without_nulls(self, data_without_nulls):
        """Test that interpolated function handles data without nulls correctly."""
        result = interpolated(data_without_nulls)

        assert result.equals(data_without_nulls)

    def test_interpolated_handles_empty_dataframe(self, empty_dataframe):
        """Test that interpolated function handles empty DataFrame correctly."""
        result = interpolated(empty_dataframe)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert result.columns == empty_dataframe.columns

    def test_interpolated_handles_single_row(self, single_row_data):
        """Test that interpolated function handles single row DataFrame."""
        result = interpolated(single_row_data)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        assert result["open"][0] is None

    def test_interpolated_interpolates_per_symbol_group(self, data_with_nulls):
        """Test that interpolation is performed separately for each symbol."""
        result = interpolated(data_with_nulls)

        aapl_result = result.filter(pl.col("symbol") == "AAPL")
        googl_result = result.filter(pl.col("symbol") == "GOOGL")

        assert len(aapl_result) == 4
        assert len(googl_result) == 3

        aapl_opens = aapl_result["open"].to_list()
        googl_opens = googl_result["open"].to_list()

        assert 150.0 in aapl_opens and 153.0 in aapl_opens
        assert 100.0 in googl_opens and 102.0 in googl_opens

    def test_interpolated_handles_all_nulls_in_column_per_symbol(self, data_all_nulls_per_symbol):
        """Test interpolated behavior when all values in a column are null for a symbol."""
        result = interpolated(data_all_nulls_per_symbol)

        aapl_result = result.filter(pl.col("symbol") == "AAPL")
        googl_result = result.filter(pl.col("symbol") == "GOOGL")

        aapl_opens = aapl_result["open"].to_list()
        googl_closes = googl_result["close"].to_list()

        assert all(val is None for val in aapl_opens)
        assert all(val is None for val in googl_closes)

    def test_interpolated_preserves_all_columns(self, data_with_nulls):
        """Test that interpolated function preserves all columns from input."""
        result = interpolated(data_with_nulls)

        assert result.columns == data_with_nulls.columns
        assert len(result) == len(data_with_nulls)

    def test_interpolated_returns_new_dataframe(self, data_with_nulls):
        """Test that interpolated function returns a new DataFrame object."""
        result = interpolated(data_with_nulls)

        assert result is not data_with_nulls
        assert isinstance(result, pl.DataFrame)

    def test_interpolated_handles_edge_nulls(self, data_first_last_nulls):
        """Test interpolated behavior with nulls at the beginning and end of sequences."""
        result = interpolated(data_first_last_nulls)

        opens = result["open"].to_list()
        closes = result["close"].to_list()
        volumes = result["volume"].to_list()

        # Boundary nulls should remain None as interpolation only works between non-null values
        assert opens[0] is None  # First value null, should stay null
        assert opens[-1] is None  # Last value null, should stay null

        assert closes[0] is None  # First value null, should stay null
        assert closes[-1] is None  # Last value null, should stay null

        assert volumes[0] is None  # First value null, should stay null
        assert volumes[-1] is None  # Last value null, should stay null

    def test_interpolated_with_mixed_data_types(self):
        """Test interpolated function with various data types including some that shouldn't be interpolated."""
        mixed_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                "open": [150.0, None, 152.0],
                "close": [154.0, None, 156.0],
                "volume": [1000000, None, 1200000],
                "some_text": ["a", None, "c"],
            }
        )

        result = interpolated(mixed_data)

        assert result["open"][1] == 151.0
        assert result["close"][1] == 155.0
        assert result["volume"][1] == 1100000

    def test_interpolated_multiple_symbols_independence(self):
        """Test that interpolation for different symbols is independent."""
        multi_symbol_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "GOOGL", "GOOGL", "MSFT", "MSFT"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                ],
                "price": [100.0, None, 200.0, None, 300.0, None],
            }
        )

        result = interpolated(multi_symbol_data)

        aapl_price = result.filter(pl.col("symbol") == "AAPL")["price"].to_list()
        googl_price = result.filter(pl.col("symbol") == "GOOGL")["price"].to_list()
        msft_price = result.filter(pl.col("symbol") == "MSFT")["price"].to_list()

        # Since each symbol only has 2 values with the second being null,
        # interpolation cannot happen (boundary nulls remain null)
        assert aapl_price[1] is None
        assert googl_price[1] is None
        assert msft_price[1] is None


class TestIntegrationBaseAndInterpolated:
    """Integration tests for base and interpolated functions used together."""

    @pytest.fixture
    def unsorted_data_with_nulls(self):
        """Create unsorted data with nulls for integration testing."""
        return pl.DataFrame(
            {
                "symbol": ["GOOGL", "AAPL", "GOOGL", "AAPL", "AAPL"],
                "date": [
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 1),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                ],
                "open": [None, 152.0, 100.0, 150.0, None],
                "close": [102.0, 156.0, None, 154.0, 155.0],
            }
        )

    def test_base_then_interpolated_integration(self, unsorted_data_with_nulls):
        """Test using base and interpolated functions in sequence."""
        sorted_data = base(unsorted_data_with_nulls)
        interpolated_data = interpolated(sorted_data)

        expected_symbol_order = ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL"]
        expected_date_order = [
            date(2023, 1, 1),
            date(2023, 1, 2),
            date(2023, 1, 3),
            date(2023, 1, 1),
            date(2023, 1, 2),
        ]

        actual_symbols = interpolated_data["symbol"].to_list()
        actual_dates = interpolated_data["date"].to_list()

        assert actual_symbols == expected_symbol_order
        assert actual_dates == expected_date_order

        aapl_data = interpolated_data.filter(pl.col("symbol") == "AAPL")
        assert aapl_data["open"][1] == 151.0

        googl_data = interpolated_data.filter(pl.col("symbol") == "GOOGL")
        assert googl_data["close"][1] == 102.0

    def test_functions_are_idempotent_when_appropriate(self, unsorted_data_with_nulls):
        """Test that applying functions multiple times gives expected results."""
        first_pass = interpolated(base(unsorted_data_with_nulls))
        second_pass = interpolated(base(first_pass))

        assert first_pass.equals(second_pass)


class TestDfpFunction:
    """Test suite for the dfp function."""

    @pytest.fixture
    def data_with_boundary_nulls(self):
        """Create data with boundary nulls (typical after interpolation)."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 4),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                ],
                "price": [None, 100.0, 102.0, None, None, 200.0, None],
                "volume": [None, 1000, 1100, None, None, 2000, None],
            }
        )

    @pytest.fixture
    def data_no_nulls(self):
        """Create data without any nulls."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                "price": [100.0, 101.0, 102.0],
                "volume": [1000, 1100, 1200],
            }
        )

    @pytest.fixture
    def data_all_nulls(self):
        """Create data where all rows have nulls."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                "price": [None, None, None],
                "volume": [1000, 1100, 1200],
            }
        )

    @pytest.fixture
    def data_mixed_nulls(self):
        """Create data with nulls scattered throughout (not just boundaries)."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 4),
                    date(2023, 1, 5),
                ],
                "price": [100.0, None, 102.0, 103.0, None],
                "volume": [1000, 1100, None, 1300, 1400],
            }
        )

    @pytest.fixture
    def empty_dataframe(self):
        """Create an empty DataFrame."""
        return pl.DataFrame(
            {
                "symbol": [],
                "date": [],
                "price": [],
                "volume": [],
            }
        )

    def test_dfp_removes_boundary_nulls(self, data_with_boundary_nulls):
        """Test that dfp function removes rows with boundary nulls."""
        result = dfp(data_with_boundary_nulls)

        # Should only keep rows without any nulls
        expected_symbols = ["AAPL", "AAPL", "GOOGL"]
        expected_dates = [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 2)]
        expected_prices = [100.0, 102.0, 200.0]
        expected_volumes = [1000, 1100, 2000]

        assert len(result) == 3
        assert result["symbol"].to_list() == expected_symbols
        assert result["date"].to_list() == expected_dates
        assert result["price"].to_list() == expected_prices
        assert result["volume"].to_list() == expected_volumes

    def test_dfp_preserves_data_without_nulls(self, data_no_nulls):
        """Test that dfp function preserves data when there are no nulls."""
        result = dfp(data_no_nulls)

        assert result.equals(data_no_nulls)
        assert len(result) == 3

    def test_dfp_handles_all_nulls_data(self, data_all_nulls):
        """Test that dfp function handles data where all rows have nulls."""
        result = dfp(data_all_nulls)

        # Should remove all rows since every row has at least one null
        assert len(result) == 0
        assert result.columns == data_all_nulls.columns

    def test_dfp_handles_mixed_nulls(self, data_mixed_nulls):
        """Test that dfp function handles scattered nulls correctly."""
        result = dfp(data_mixed_nulls)

        # Should keep only rows with no nulls at all
        expected_rows = 2  # Only rows at index 0 and 3 have no nulls
        assert len(result) == expected_rows

        # Verify the remaining rows
        expected_prices = [100.0, 103.0]
        expected_volumes = [1000, 1300]
        assert result["price"].to_list() == expected_prices
        assert result["volume"].to_list() == expected_volumes

    def test_dfp_handles_empty_dataframe(self, empty_dataframe):
        """Test that dfp function handles empty DataFrame."""
        result = dfp(empty_dataframe)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert result.columns == empty_dataframe.columns

    def test_dfp_preserves_column_types(self, data_with_boundary_nulls):
        """Test that dfp function preserves column data types."""
        result = dfp(data_with_boundary_nulls)

        original_schema = data_with_boundary_nulls.schema
        result_schema = result.schema

        assert result_schema == original_schema

    def test_dfp_preserves_column_order(self, data_with_boundary_nulls):
        """Test that dfp function preserves column order."""
        result = dfp(data_with_boundary_nulls)

        assert result.columns == data_with_boundary_nulls.columns

    def test_dfp_returns_new_dataframe(self, data_no_nulls):
        """Test that dfp function returns a new DataFrame object."""
        result = dfp(data_no_nulls)

        assert result is not data_no_nulls
        assert isinstance(result, pl.DataFrame)

    def test_dfp_maintains_sorting_order(self):
        """Test that dfp function maintains the sorted order of data."""
        # Create sorted data with some nulls
        sorted_data_with_nulls = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                ],
                "price": [None, 100.0, 102.0, 200.0, None, 202.0],
            }
        )

        result = dfp(sorted_data_with_nulls)

        # Verify that the remaining data is still sorted
        result_symbols = result["symbol"].to_list()
        result_dates = result["date"].to_list()

        # Should maintain AAPL before GOOGL, and dates in ascending order within each symbol
        expected_order = [
            ("AAPL", date(2023, 1, 2)),
            ("AAPL", date(2023, 1, 3)),
            ("GOOGL", date(2023, 1, 1)),
            ("GOOGL", date(2023, 1, 3)),
        ]

        actual_order = list(zip(result_symbols, result_dates, strict=False))
        assert actual_order == expected_order

    def test_dfp_handles_single_row_with_null(self):
        """Test dfp function with single row containing null."""
        single_row_null = pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [date(2023, 1, 1)],
                "price": [None],
            }
        )

        result = dfp(single_row_null)

        assert len(result) == 0
        assert result.columns == single_row_null.columns

    def test_dfp_handles_single_row_without_null(self):
        """Test dfp function with single row without null."""
        single_row_no_null = pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [date(2023, 1, 1)],
                "price": [100.0],
            }
        )

        result = dfp(single_row_no_null)

        assert result.equals(single_row_no_null)
        assert len(result) == 1

    def test_dfp_with_multiple_symbol_groups(self):
        """Test that dfp function works correctly across multiple symbol groups."""
        multi_symbol_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "GOOGL", "GOOGL", "MSFT", "MSFT"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                ],
                "price": [None, 100.0, 200.0, None, None, None],
                "volume": [None, 1000, 2000, None, None, None],
            }
        )

        result = dfp(multi_symbol_data)

        # Should keep only the complete rows
        assert len(result) == 2
        expected_symbols = ["AAPL", "GOOGL"]
        expected_prices = [100.0, 200.0]

        assert result["symbol"].to_list() == expected_symbols
        assert result["price"].to_list() == expected_prices


class TestIntegrationAllFunctions:
    """Integration tests for base, interpolated, and dfp functions used together."""

    @pytest.fixture
    def raw_unsorted_data_with_gaps(self):
        """Create raw unsorted data with missing values and gaps."""
        return pl.DataFrame(
            {
                "symbol": ["GOOGL", "AAPL", "GOOGL", "AAPL", "AAPL", "GOOGL"],
                "date": [
                    date(2023, 1, 2),
                    date(2023, 1, 4),
                    date(2023, 1, 1),
                    date(2023, 1, 1),
                    date(2023, 1, 3),
                    date(2023, 1, 3),
                ],
                "price": [None, 104.0, 100.0, 100.0, None, 102.0],
                "volume": [None, 1400, 1000, 1000, 1200, None],
            }
        )

    def test_complete_preprocessing_pipeline(self, raw_unsorted_data_with_gaps):
        """Test the complete preprocessing pipeline: base -> interpolated -> dfp."""
        # Step 1: Sort the data
        sorted_data = base(raw_unsorted_data_with_gaps)

        # Step 2: Interpolate missing values
        interpolated_data = interpolated(sorted_data)

        # Step 3: dfp boundary nulls
        dfped_data = dfp(interpolated_data)

        # Verify the pipeline worked correctly
        assert len(dfped_data) > 0  # Should have some data remaining
        assert dfped_data.null_count().sum_horizontal()[0] == 0  # No nulls should remain

        # Verify data is still sorted
        symbols = dfped_data["symbol"].to_list()
        dates = dfped_data["date"].to_list()

        # Check that symbols are in order and dates within symbols are ascending
        for i in range(len(symbols) - 1):
            if symbols[i] == symbols[i + 1]:
                assert dates[i] <= dates[i + 1]
            else:
                assert symbols[i] <= symbols[i + 1]

    def test_pipeline_functions_are_composable(self, raw_unsorted_data_with_gaps):
        """Test that the functions can be composed together."""
        # Single composition
        result = dfp(interpolated(base(raw_unsorted_data_with_gaps)))

        assert isinstance(result, pl.DataFrame)
        assert result.null_count().sum_horizontal()[0] == 0  # No nulls

    def test_pipeline_with_all_valid_data(self):
        """Test pipeline with data that has no nulls or sorting issues."""
        perfect_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "GOOGL", "GOOGL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                ],
                "price": [100.0, 101.0, 200.0, 201.0],
                "volume": [1000, 1100, 2000, 2100],
            }
        )

        result = dfp(interpolated(base(perfect_data)))

        # Should preserve all data since it's already perfect
        assert len(result) == 4
        assert result.equals(perfect_data)

    def test_pipeline_with_edge_cases(self):
        """Test pipeline with various edge cases."""
        # Data where some symbols have all nulls, others have valid data
        edge_case_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "GOOGL", "GOOGL", "MSFT"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 1),
                ],
                "price": [100.0, 102.0, None, None, None],  # GOOGL & MSFT have nulls
                "volume": [1000, 1200, None, None, None],
            }
        )

        result = dfp(interpolated(base(edge_case_data)))

        # Should only keep AAPL data
        assert len(result) == 2
        assert all(symbol == "AAPL" for symbol in result["symbol"].to_list())


class TestLogReturnsDFunction:
    """Test suite for the log_returns_d function."""

    @pytest.fixture
    def basic_price_data(self):
        """Create basic data with close prices for log returns calculation."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 4),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                ],
                "close": [100.0, 105.0, 102.0, 110.0, 200.0, 210.0, 205.0],
            }
        )

    @pytest.fixture
    def single_symbol_data(self):
        """Create data with single symbol."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                "close": [100.0, 110.0, 121.0],
            }
        )

    @pytest.fixture
    def single_row_data(self):
        """Create data with single row."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [date(2023, 1, 1)],
                "close": [100.0],
            }
        )

    @pytest.fixture
    def empty_dataframe(self):
        """Create empty DataFrame."""
        return pl.DataFrame(
            {
                "symbol": [],
                "date": [],
                "close": [],
            }
        )

    def test_log_returns_d_calculates_correctly(self, basic_price_data):
        """Test that log_returns_d calculates log returns correctly within each symbol group."""
        import math
        
        result = log_returns_d(basic_price_data)
        
        # Check that log_returns_d column was added
        assert "log_returns_d" in result.columns
        
        aapl_data = result.filter(pl.col("symbol") == "AAPL")
        googl_data = result.filter(pl.col("symbol") == "GOOGL")
        
        # For AAPL: 100.0 -> 105.0 -> 102.0 -> 110.0
        # First return should be None (no previous value)
        assert aapl_data["log_returns_d"][0] is None
        
        # Second return: log(105.0) - log(100.0)
        expected_return_1 = math.log(105.0) - math.log(100.0)
        assert abs(aapl_data["log_returns_d"][1] - expected_return_1) < 1e-10
        
        # Third return: log(102.0) - log(105.0) 
        expected_return_2 = math.log(102.0) - math.log(105.0)
        assert abs(aapl_data["log_returns_d"][2] - expected_return_2) < 1e-10
        
        # For GOOGL: first return should be None
        assert googl_data["log_returns_d"][0] is None

    def test_log_returns_d_per_symbol_group(self, basic_price_data):
        """Test that log returns are calculated separately for each symbol."""
        result = log_returns_d(basic_price_data)
        
        aapl_data = result.filter(pl.col("symbol") == "AAPL")
        googl_data = result.filter(pl.col("symbol") == "GOOGL")
        
        # Each symbol should have its own first null value
        assert aapl_data["log_returns_d"][0] is None
        assert googl_data["log_returns_d"][0] is None
        
        # Subsequent values should be calculated within each group
        assert aapl_data["log_returns_d"][1] is not None
        assert googl_data["log_returns_d"][1] is not None

    def test_log_returns_d_preserves_other_columns(self, basic_price_data):
        """Test that log_returns_d preserves all original columns."""
        result = log_returns_d(basic_price_data)
        
        # Check all original columns are preserved
        for col in basic_price_data.columns:
            assert col in result.columns
            
        # Check that data in original columns is unchanged
        assert result["symbol"].equals(basic_price_data["symbol"])
        assert result["date"].equals(basic_price_data["date"])
        assert result["close"].equals(basic_price_data["close"])

    def test_log_returns_d_handles_single_symbol(self, single_symbol_data):
        """Test log_returns_d with single symbol."""
        import math
        
        result = log_returns_d(single_symbol_data)
        
        assert len(result) == 3
        assert result["log_returns_d"][0] is None
        
        # Check the calculations
        expected_return_1 = math.log(110.0) - math.log(100.0)
        expected_return_2 = math.log(121.0) - math.log(110.0)
        
        assert abs(result["log_returns_d"][1] - expected_return_1) < 1e-10
        assert abs(result["log_returns_d"][2] - expected_return_2) < 1e-10

    def test_log_returns_d_handles_single_row(self, single_row_data):
        """Test log_returns_d with single row."""
        result = log_returns_d(single_row_data)
        
        assert len(result) == 1
        assert result["log_returns_d"][0] is None

    def test_log_returns_d_handles_empty_dataframe(self, empty_dataframe):
        """Test log_returns_d with empty DataFrame."""
        result = log_returns_d(empty_dataframe)
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert "log_returns_d" in result.columns

    def test_log_returns_d_returns_new_dataframe(self, basic_price_data):
        """Test that log_returns_d returns a new DataFrame object."""
        result = log_returns_d(basic_price_data)
        
        assert result is not basic_price_data
        assert isinstance(result, pl.DataFrame)

    def test_log_returns_d_with_identical_prices(self):
        """Test log_returns_d behavior with identical consecutive prices."""
        identical_prices_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                "close": [100.0, 100.0, 100.0],
            }
        )
        
        result = log_returns_d(identical_prices_data)
        
        # Log returns should be 0.0 for identical prices
        assert result["log_returns_d"][0] is None
        assert result["log_returns_d"][1] == 0.0
        assert result["log_returns_d"][2] == 0.0


class TestDollarVolumeFunction:
    """Test suite for the dollar_volume function."""

    @pytest.fixture
    def basic_trading_data(self):
        """Create basic data with close prices and volume."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                ],
                "close": [100.0, 105.0, 102.0, 200.0, 210.0],
                "volume": [1000, 1500, 1200, 2000, 2500],
            }
        )

    @pytest.fixture
    def zero_values_data(self):
        """Create data with zero values."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                "close": [0.0, 105.0, 102.0],
                "volume": [1000, 0, 1200],
            }
        )

    @pytest.fixture
    def single_row_data(self):
        """Create data with single row."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [date(2023, 1, 1)],
                "close": [100.0],
                "volume": [1000],
            }
        )

    @pytest.fixture
    def empty_dataframe(self):
        """Create empty DataFrame."""
        return pl.DataFrame(
            {
                "symbol": [],
                "date": [],
                "close": [],
                "volume": [],
            }
        )

    def test_dollar_volume_calculates_correctly(self, basic_trading_data):
        """Test that dollar_volume calculates correctly."""
        result = dollar_volume(basic_trading_data)
        
        # Check that dollar_volume column was added
        assert "dollar_volume" in result.columns
        
        # Check calculations
        expected_values = [
            100.0 * 1000,  # 100000.0
            105.0 * 1500,  # 157500.0
            102.0 * 1200,  # 122400.0
            200.0 * 2000,  # 400000.0
            210.0 * 2500,  # 525000.0
        ]
        
        actual_values = result["dollar_volume"].to_list()
        
        for expected, actual in zip(expected_values, actual_values):
            assert abs(expected - actual) < 1e-6

    def test_dollar_volume_preserves_other_columns(self, basic_trading_data):
        """Test that dollar_volume preserves all original columns."""
        result = dollar_volume(basic_trading_data)
        
        # Check all original columns are preserved
        for col in basic_trading_data.columns:
            assert col in result.columns
            
        # Check that data in original columns is unchanged
        assert result["symbol"].equals(basic_trading_data["symbol"])
        assert result["date"].equals(basic_trading_data["date"])
        assert result["close"].equals(basic_trading_data["close"])
        assert result["volume"].equals(basic_trading_data["volume"])

    def test_dollar_volume_handles_zero_values(self, zero_values_data):
        """Test dollar_volume with zero values."""
        result = dollar_volume(zero_values_data)
        
        expected_values = [
            0.0 * 1000,   # 0.0
            105.0 * 0,    # 0.0
            102.0 * 1200, # 122400.0
        ]
        
        actual_values = result["dollar_volume"].to_list()
        
        for expected, actual in zip(expected_values, actual_values):
            assert abs(expected - actual) < 1e-6

    def test_dollar_volume_handles_single_row(self, single_row_data):
        """Test dollar_volume with single row."""
        result = dollar_volume(single_row_data)
        
        assert len(result) == 1
        assert result["dollar_volume"][0] == 100.0 * 1000

    def test_dollar_volume_handles_empty_dataframe(self, empty_dataframe):
        """Test dollar_volume with empty DataFrame."""
        result = dollar_volume(empty_dataframe)
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert "dollar_volume" in result.columns

    def test_dollar_volume_returns_new_dataframe(self, basic_trading_data):
        """Test that dollar_volume returns a new DataFrame object."""
        result = dollar_volume(basic_trading_data)
        
        assert result is not basic_trading_data
        assert isinstance(result, pl.DataFrame)

    def test_dollar_volume_with_fractional_values(self):
        """Test dollar_volume with fractional prices and volumes."""
        fractional_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "date": [date(2023, 1, 1), date(2023, 1, 2)],
                "close": [100.55, 105.75],
                "volume": [1000, 1500],
            }
        )
        
        result = dollar_volume(fractional_data)
        
        expected_values = [
            100.55 * 1000,  # 100550.0
            105.75 * 1500,  # 158625.0
        ]
        
        actual_values = result["dollar_volume"].to_list()
        
        for expected, actual in zip(expected_values, actual_values):
            assert abs(expected - actual) < 1e-6


class TestYLogReturnsFunction:
    """Test suite for the y_log_returns function."""

    @pytest.fixture
    def basic_close_data(self):
        """Create data with close prices for testing y_log_returns."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL"],
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 4),
                    date(2023, 1, 5),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                ],
                "close": [100.0, 105.0, 102.0, 110.0, 108.0, 200.0, 210.0, 205.0],
            }
        )

    @pytest.fixture
    def single_symbol_data(self):
        """Create data with single symbol."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
                "close": [100.0, 110.0, 121.0, 105.0],
            }
        )

    @pytest.fixture
    def short_data(self):
        """Create data shorter than horizon window."""
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "date": [date(2023, 1, 1), date(2023, 1, 2)],
                "close": [100.0, 105.0],
            }
        )

    @pytest.fixture
    def empty_dataframe(self):
        """Create empty DataFrame."""
        return pl.DataFrame(
            {
                "symbol": [],
                "date": [],
                "close": [],
            }
        )

    def test_y_log_returns_calculates_correctly(self, basic_close_data):
        """Test that y_log_returns calculates correctly with horizon=2."""
        import math
        
        result = y_log_returns(basic_close_data, horizon=2)
        
        # Check that y_log_returns column was added
        assert "y_log_returns" in result.columns
        
        aapl_data = result.filter(pl.col("symbol") == "AAPL")
        googl_data = result.filter(pl.col("symbol") == "GOOGL")
        
        # For AAPL with horizon=2:
        # Row 0: ln(102.0) - ln(100.0) = ln(102.0/100.0)
        expected_0 = math.log(102.0) - math.log(100.0)
        assert abs(aapl_data["y_log_returns"][0] - expected_0) < 1e-10
        
        # Row 1: ln(110.0) - ln(105.0) = ln(110.0/105.0)
        expected_1 = math.log(110.0) - math.log(105.0)
        assert abs(aapl_data["y_log_returns"][1] - expected_1) < 1e-10
        
        # Row 2: ln(108.0) - ln(102.0) = ln(108.0/102.0)
        expected_2 = math.log(108.0) - math.log(102.0)
        assert abs(aapl_data["y_log_returns"][2] - expected_2) < 1e-10
        
        # Row 3: no future value 2 steps ahead, should be None
        assert aapl_data["y_log_returns"][3] is None
        
        # Row 4: no future value 2 steps ahead, should be None  
        assert aapl_data["y_log_returns"][4] is None
        
        # For GOOGL with horizon=2:
        # Row 0: ln(205.0) - ln(200.0)
        expected_googl_0 = math.log(205.0) - math.log(200.0)
        assert abs(googl_data["y_log_returns"][0] - expected_googl_0) < 1e-10
        
        # Rows 1 and 2: should be None (no future values 2 steps ahead)
        assert googl_data["y_log_returns"][1] is None
        assert googl_data["y_log_returns"][2] is None

    def test_y_log_returns_per_symbol_group(self, basic_close_data):
        """Test that y_log_returns calculations are done separately for each symbol."""
        import math
        
        result = y_log_returns(basic_close_data, horizon=1)
        
        aapl_data = result.filter(pl.col("symbol") == "AAPL")
        googl_data = result.filter(pl.col("symbol") == "GOOGL")
        
        # Each symbol should have calculations done independently
        # AAPL row 0: ln(105.0) - ln(100.0)
        expected_aapl_0 = math.log(105.0) - math.log(100.0)
        assert abs(aapl_data["y_log_returns"][0] - expected_aapl_0) < 1e-10
        
        # AAPL row 1: ln(102.0) - ln(105.0)
        expected_aapl_1 = math.log(102.0) - math.log(105.0)
        assert abs(aapl_data["y_log_returns"][1] - expected_aapl_1) < 1e-10
        
        # GOOGL row 0: ln(210.0) - ln(200.0)
        expected_googl_0 = math.log(210.0) - math.log(200.0)
        assert abs(googl_data["y_log_returns"][0] - expected_googl_0) < 1e-10
        
        # GOOGL row 1: ln(205.0) - ln(210.0)
        expected_googl_1 = math.log(205.0) - math.log(210.0)
        assert abs(googl_data["y_log_returns"][1] - expected_googl_1) < 1e-10

    def test_y_log_returns_preserves_other_columns(self, basic_close_data):
        """Test that y_log_returns preserves all original columns."""
        result = y_log_returns(basic_close_data, horizon=2)
        
        # Check all original columns are preserved
        for col in basic_close_data.columns:
            assert col in result.columns
            
        # Check that data in original columns is unchanged
        assert result["symbol"].equals(basic_close_data["symbol"])
        assert result["date"].equals(basic_close_data["date"])
        assert result["close"].equals(basic_close_data["close"])

    def test_y_log_returns_handles_different_horizons(self, single_symbol_data):
        """Test y_log_returns with different horizon sizes."""
        import math
        
        # Test with horizon=1
        result_1 = y_log_returns(single_symbol_data, horizon=1)
        
        # Row 0: ln(110.0) - ln(100.0)
        expected_0 = math.log(110.0) - math.log(100.0)
        assert abs(result_1["y_log_returns"][0] - expected_0) < 1e-10
        
        # Row 1: ln(121.0) - ln(110.0)
        expected_1 = math.log(121.0) - math.log(110.0)
        assert abs(result_1["y_log_returns"][1] - expected_1) < 1e-10
        
        # Row 2: ln(105.0) - ln(121.0)
        expected_2 = math.log(105.0) - math.log(121.0)
        assert abs(result_1["y_log_returns"][2] - expected_2) < 1e-10
        
        # Row 3: no future value, should be None
        assert result_1["y_log_returns"][3] is None
        
        # Test with horizon=3
        result_3 = y_log_returns(single_symbol_data, horizon=3)
        
        # Only row 0 can have a calculation: ln(105.0) - ln(100.0)
        expected_0_h3 = math.log(105.0) - math.log(100.0)
        assert abs(result_3["y_log_returns"][0] - expected_0_h3) < 1e-10
        
        # All other rows should be None (no future values 3 steps ahead)
        for i in range(1, 4):
            assert result_3["y_log_returns"][i] is None

    def test_y_log_returns_handles_short_data(self, short_data):
        """Test y_log_returns when data is shorter than horizon window."""
        result = y_log_returns(short_data, horizon=5)
        
        # All values should be None since horizon exceeds data length
        assert all(val is None for val in result["y_log_returns"])

    def test_y_log_returns_handles_empty_dataframe(self, empty_dataframe):
        """Test y_log_returns with empty DataFrame."""
        result = y_log_returns(empty_dataframe, horizon=2)
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert "y_log_returns" in result.columns

    def test_y_log_returns_returns_new_dataframe(self, basic_close_data):
        """Test that y_log_returns returns a new DataFrame object."""
        result = y_log_returns(basic_close_data, horizon=2)
        
        assert result is not basic_close_data
        assert isinstance(result, pl.DataFrame)

    def test_y_log_returns_with_zero_horizon(self, single_symbol_data):
        """Test y_log_returns with horizon=0."""
        result = y_log_returns(single_symbol_data, horizon=0)
        
        # With horizon=0, each row should be: ln(current) - ln(current) = 0
        assert result["y_log_returns"][0] == 0.0   # ln(100.0) - ln(100.0) = 0.0
        assert result["y_log_returns"][1] == 0.0   # ln(110.0) - ln(110.0) = 0.0
        assert result["y_log_returns"][2] == 0.0   # ln(121.0) - ln(121.0) = 0.0
        assert result["y_log_returns"][3] == 0.0   # ln(105.0) - ln(105.0) = 0.0

    def test_y_log_returns_with_identical_prices(self):
        """Test y_log_returns behavior with identical consecutive prices."""
        identical_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
                "close": [100.0, 100.0, 100.0, 100.0],
            }
        )
        
        result = y_log_returns(identical_data, horizon=1)
        
        # Log returns should be 0.0 for identical prices
        assert result["y_log_returns"][0] == 0.0  # ln(100.0) - ln(100.0) = 0.0
        assert result["y_log_returns"][1] == 0.0  # ln(100.0) - ln(100.0) = 0.0
        assert result["y_log_returns"][2] == 0.0  # ln(100.0) - ln(100.0) = 0.0
        assert result["y_log_returns"][3] is None  # No future value

    def test_y_log_returns_with_multiple_symbols_independence(self):
        """Test that y_log_returns calculations for different symbols are independent."""
        multi_symbol_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL", "MSFT", "MSFT"],
                "date": [
                    date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3),
                    date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3),
                    date(2023, 1, 1), date(2023, 1, 2),
                ],
                "close": [100.0, 110.0, 121.0, 200.0, 220.0, 242.0, 300.0, 330.0],
            }
        )

        result = y_log_returns(multi_symbol_data, horizon=1)

        aapl_results = result.filter(pl.col("symbol") == "AAPL")["y_log_returns"].to_list()
        googl_results = result.filter(pl.col("symbol") == "GOOGL")["y_log_returns"].to_list()
        msft_results = result.filter(pl.col("symbol") == "MSFT")["y_log_returns"].to_list()

        # Each symbol should have independent calculations
        import math
        
        # AAPL: ln(110.0/100.0), ln(121.0/110.0), None
        assert abs(aapl_results[0] - math.log(110.0/100.0)) < 1e-10
        assert abs(aapl_results[1] - math.log(121.0/110.0)) < 1e-10
        assert aapl_results[2] is None
        
        # GOOGL: ln(220.0/200.0), ln(242.0/220.0), None
        assert abs(googl_results[0] - math.log(220.0/200.0)) < 1e-10
        assert abs(googl_results[1] - math.log(242.0/220.0)) < 1e-10
        assert googl_results[2] is None
        
        # MSFT: ln(330.0/300.0), None
        assert abs(msft_results[0] - math.log(330.0/300.0)) < 1e-10
        assert msft_results[1] is None

    def test_y_log_returns_handles_single_row_per_symbol(self):
        """Test y_log_returns with single row per symbol."""
        single_row_data = pl.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL"],
                "date": [date(2023, 1, 1), date(2023, 1, 1)],
                "close": [100.0, 200.0],
            }
        )
        
        result = y_log_returns(single_row_data, horizon=1)
        
        # All values should be None since there are no future values
        assert all(val is None for val in result["y_log_returns"])
        assert len(result) == 2


class TestDffFunction:
    """Test suite for the dff function."""

    @pytest.fixture
    def log_returns_data(self):
        """Create sample data with daily log returns."""
        return pl.DataFrame(
            {
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                ],
                "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL"],
                "log_returns_d": [None, 0.01, 0.02, None, 0.03],
                "other_col": ["a", "b", "c", "d", "e"],
            }
        )

    @pytest.fixture
    def y_log_returns_data(self):
        """Create sample data with forward log returns."""
        return pl.DataFrame(
            {
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                ],
                "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL"],
                "y_log_returns": [0.05, 0.03, None, 0.08, None],
                "extra_col": [1, 2, 3, 4, 5],
            }
        )

    @pytest.fixture
    def dollar_volume_data(self):
        """Create sample data with dollar volume and OHLC data."""
        return pl.DataFrame(
            {
                "date": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                ],
                "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL"],
                "dollar_volume": [100000.0, 157500.0, 122400.0, 400000.0, 525000.0],
                "open": [100.0, 105.0, 102.0, 200.0, 210.0],
                "high": [102.0, 107.0, 104.0, 202.0, 212.0],
                "low": [99.0, 104.0, 101.0, 199.0, 209.0],
                "close": [100.0, 105.0, 102.0, 200.0, 210.0],
                "volume": [1000, 1500, 1200, 2000, 2500],
                "unused_col": ["x", "y", "z", "w", "v"],
            }
        )

    @pytest.fixture
    def partial_overlap_data(self):
        """Create data where not all date-symbol pairs overlap across inputs."""
        log_returns = pl.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "log_returns_d": [None, 0.01, 0.02],
            }
        )
        
        y_returns = pl.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2)],  # Missing day 3
                "symbol": ["AAPL", "AAPL"],
                "y_log_returns": [0.05, 0.03],
            }
        )
        
        dollar_vol = pl.DataFrame(
            {
                "date": [date(2023, 1, 2), date(2023, 1, 3)],  # Missing day 1
                "symbol": ["AAPL", "AAPL"],
                "dollar_volume": [157500.0, 122400.0],
                "open": [105.0, 102.0],
                "high": [107.0, 104.0],
                "low": [104.0, 101.0],
                "close": [105.0, 102.0],
                "volume": [1500, 1200],
            }
        )
        
        return log_returns, y_returns, dollar_vol

    @pytest.fixture
    def empty_dataframes(self):
        """Create empty DataFrames."""
        empty_log_returns = pl.DataFrame(
            {
                "date": [],
                "symbol": [],
                "log_returns_d": [],
            }
        )
        
        empty_y_returns = pl.DataFrame(
            {
                "date": [],
                "symbol": [],
                "y_log_returns": [],
            }
        )
        
        empty_dollar_vol = pl.DataFrame(
            {
                "date": [],
                "symbol": [],
                "dollar_volume": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        )
        
        return empty_log_returns, empty_y_returns, empty_dollar_vol

    def test_dff_merges_correctly(
        self, log_returns_data, y_log_returns_data, dollar_volume_data
    ):
        """Test that dff function merges DataFrames correctly."""
        result = dff(log_returns_data, y_log_returns_data, dollar_volume_data)
        
        # Check that all expected columns are present
        expected_columns = [
            "date",
            "symbol",
            "log_returns_d",
            "y_log_returns",
            "dollar_volume",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        
        for col in expected_columns:
            assert col in result.columns
        
        # Check that only the selected columns are present (no extra columns)
        assert len(result.columns) == len(expected_columns)
        
        # Check that the merge worked correctly
        assert len(result) == 5  # All rows from input should match

    def test_dff_preserves_data_integrity(
        self, log_returns_data, y_log_returns_data, dollar_volume_data
    ):
        """Test that dff preserves data integrity during merge."""
        result = dff(log_returns_data, y_log_returns_data, dollar_volume_data)
        
        # Check first few rows to ensure data integrity
        aapl_result = result.filter(pl.col("symbol") == "AAPL")
        googl_result = result.filter(pl.col("symbol") == "GOOGL")
        
        assert len(aapl_result) == 3
        assert len(googl_result) == 2
        
        # Check that values are correctly merged
        # AAPL row 1 (index 0 after filtering)
        aapl_row_1 = aapl_result[0]
        assert aapl_row_1["log_returns_d"][0] is None  # From log_returns_data
        assert aapl_row_1["y_log_returns"][0] == 0.05  # From y_log_returns_data
        assert aapl_row_1["dollar_volume"][0] == 100000.0  # From dollar_volume_data
        assert aapl_row_1["open"][0] == 100.0  # From dollar_volume_data
        
        # GOOGL row 1 (index 0 after filtering)
        googl_row_1 = googl_result[0]
        assert googl_row_1["log_returns_d"][0] is None
        assert googl_row_1["y_log_returns"][0] == 0.08
        assert googl_row_1["dollar_volume"][0] == 400000.0

    def test_dff_sorts_output(
        self, log_returns_data, y_log_returns_data, dollar_volume_data
    ):
        """Test that dff sorts output by symbol and date."""
        result = dff(log_returns_data, y_log_returns_data, dollar_volume_data)
        
        # Check sorting
        symbols = result["symbol"].to_list()
        dates = result["date"].to_list()
        
        # Verify symbol order (AAPL should come before GOOGL)
        aapl_indices = [i for i, s in enumerate(symbols) if s == "AAPL"]
        googl_indices = [i for i, s in enumerate(symbols) if s == "GOOGL"]
        
        assert max(aapl_indices) < min(googl_indices)
        
        # Verify date order within each symbol
        aapl_dates = [dates[i] for i in aapl_indices]
        googl_dates = [dates[i] for i in googl_indices]
        
        assert aapl_dates == sorted(aapl_dates)
        assert googl_dates == sorted(googl_dates)

    def test_dff_handles_partial_overlap(self, partial_overlap_data):
        """Test that dff handles cases where not all date-symbol pairs overlap."""
        log_returns, y_returns, dollar_vol = partial_overlap_data
        
        result = dff(log_returns, y_returns, dollar_vol)
        
        # Only date 2023-01-02 should be present (only overlap)
        assert len(result) == 1
        assert result["date"][0] == date(2023, 1, 2)
        assert result["symbol"][0] == "AAPL"
        
        # Check that merged data is correct
        assert result["log_returns_d"][0] == 0.01  # From log_returns
        assert result["y_log_returns"][0] == 0.03   # From y_returns  
        assert result["dollar_volume"][0] == 157500.0  # From dollar_vol

    def test_dff_handles_empty_dataframes(self, empty_dataframes):
        """Test that dff handles empty input DataFrames."""
        empty_log_returns, empty_y_returns, empty_dollar_vol = empty_dataframes
        
        result = dff(empty_log_returns, empty_y_returns, empty_dollar_vol)
        
        # Should return empty DataFrame with correct structure
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        
        expected_columns = [
            "date",
            "symbol", 
            "log_returns_d",
            "y_log_returns",
            "dollar_volume",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        
        for col in expected_columns:
            assert col in result.columns

    def test_dff_returns_new_dataframe(
        self, log_returns_data, y_log_returns_data, dollar_volume_data
    ):
        """Test that dff returns a new DataFrame object."""
        result = dff(log_returns_data, y_log_returns_data, dollar_volume_data)
        
        # Should be different objects
        assert result is not log_returns_data
        assert result is not y_log_returns_data
        assert result is not dollar_volume_data
        assert isinstance(result, pl.DataFrame)

    def test_dff_ignores_extra_columns(
        self, log_returns_data, y_log_returns_data, dollar_volume_data
    ):
        """Test that dff only selects the expected columns, ignoring extras."""
        result = dff(log_returns_data, y_log_returns_data, dollar_volume_data)
        
        # Extra columns should not be in result
        assert "other_col" not in result.columns  # From log_returns_data
        assert "extra_col" not in result.columns  # From y_log_returns_data
        assert "unused_col" not in result.columns  # From dollar_volume_data

    def test_dff_with_single_row_per_symbol(self):
        """Test dff with single row per symbol."""
        log_returns_single = pl.DataFrame(
            {
                "date": [date(2023, 1, 1)],
                "symbol": ["AAPL"],
                "log_returns_d": [None],
            }
        )
        
        y_returns_single = pl.DataFrame(
            {
                "date": [date(2023, 1, 1)],
                "symbol": ["AAPL"],
                "y_log_returns": [0.05],
            }
        )
        
        dollar_vol_single = pl.DataFrame(
            {
                "date": [date(2023, 1, 1)],
                "symbol": ["AAPL"],
                "dollar_volume": [100000.0],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000],
            }
        )
        
        result = dff(log_returns_single, y_returns_single, dollar_vol_single)
        
        assert len(result) == 1
        assert result["symbol"][0] == "AAPL"
        assert result["log_returns_d"][0] is None
        assert result["y_log_returns"][0] == 0.05
        assert result["dollar_volume"][0] == 100000.0

    def test_dff_with_multiple_symbols_complex(self):
        """Test dff with multiple symbols and complex overlap scenarios."""
        log_returns_multi = pl.DataFrame(
            {
                "date": [
                    date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3),
                    date(2023, 1, 1), date(2023, 1, 2),
                    date(2023, 1, 1)
                ],
                "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "MSFT"],
                "log_returns_d": [None, 0.01, 0.02, None, 0.03, None],
            }
        )
        
        y_returns_multi = pl.DataFrame(
            {
                "date": [
                    date(2023, 1, 1), date(2023, 1, 2),
                    date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3),
                    date(2023, 1, 1), date(2023, 1, 2)
                ],
                "symbol": ["AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL", "MSFT", "MSFT"],
                "y_log_returns": [0.05, 0.03, 0.08, None, 0.09, 0.02, 0.04],
            }
        )
        
        dollar_vol_multi = pl.DataFrame(
            {
                "date": [
                    date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3),
                    date(2023, 1, 1),
                    date(2023, 1, 2)
                ],
                "symbol": ["AAPL", "AAPL", "AAPL", "GOOGL", "MSFT"],
                "dollar_volume": [100000.0, 157500.0, 122400.0, 400000.0, 300000.0],
                "open": [100.0, 105.0, 102.0, 200.0, 300.0],
                "high": [102.0, 107.0, 104.0, 202.0, 302.0],
                "low": [99.0, 104.0, 101.0, 199.0, 299.0],
                "close": [101.0, 106.0, 103.0, 201.0, 301.0],
                "volume": [1000, 1500, 1200, 2000, 3000],
            }
        )
        
        result = dff(log_returns_multi, y_returns_multi, dollar_vol_multi)
        
        # Check that only overlapping date-symbol pairs are included
        result_pairs = [(row["symbol"], row["date"]) for row in result.iter_rows(named=True)]
        
        expected_pairs = [
            ("AAPL", date(2023, 1, 1)),
            ("AAPL", date(2023, 1, 2)), 
            ("GOOGL", date(2023, 1, 1))
        ]
        
        assert len(result) == 3
        for pair in expected_pairs:
            assert pair in result_pairs

    def test_dff_column_data_types_preserved(
        self, log_returns_data, y_log_returns_data, dollar_volume_data
    ):
        """Test that dff preserves appropriate data types."""
        result = dff(log_returns_data, y_log_returns_data, dollar_volume_data)
        
        # Check that numeric columns have appropriate types
        assert result["log_returns_d"].dtype in [pl.Float64, pl.Float32]
        assert result["y_log_returns"].dtype in [pl.Float64, pl.Float32]
        assert result["dollar_volume"].dtype in [pl.Float64, pl.Float32]
        assert result["open"].dtype in [pl.Float64, pl.Float32]
        assert result["high"].dtype in [pl.Float64, pl.Float32]
        assert result["low"].dtype in [pl.Float64, pl.Float32]
        assert result["close"].dtype in [pl.Float64, pl.Float32]
        assert result["volume"].dtype in [pl.Int64, pl.Int32]
        
        # Check that date and symbol columns have appropriate types
        assert result["date"].dtype == pl.Date
        assert result["symbol"].dtype == pl.Utf8

    def test_dff_maintains_null_values(self):
        """Test that dff properly maintains null values after merging."""
        log_returns_nulls = pl.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2)],
                "symbol": ["AAPL", "AAPL"],
                "log_returns_d": [None, 0.01],
            }
        )
        
        y_returns_nulls = pl.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2)],
                "symbol": ["AAPL", "AAPL"],
                "y_log_returns": [0.05, None],
            }
        )
        
        dollar_vol_nulls = pl.DataFrame(
            {
                "date": [date(2023, 1, 1), date(2023, 1, 2)],
                "symbol": ["AAPL", "AAPL"],
                "dollar_volume": [100000.0, 157500.0],
                "open": [100.0, 105.0],
                "high": [102.0, 107.0],
                "low": [99.0, 104.0],
                "close": [101.0, 106.0],
                "volume": [1000, 1500],
            }
        )
        
        result = dff(log_returns_nulls, y_returns_nulls, dollar_vol_nulls)
        
        # Check that nulls are preserved correctly
        assert result["log_returns_d"][0] is None  # First row log_returns_d
        assert result["y_log_returns"][1] is None  # Second row y_log_returns
        
        # Check that non-null values are preserved
        assert result["log_returns_d"][1] == 0.01
        assert result["y_log_returns"][0] == 0.05

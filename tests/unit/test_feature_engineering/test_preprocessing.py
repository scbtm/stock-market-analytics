from datetime import date

import polars as pl
import pytest

from stock_market_analytics.feature_engineering.preprocessing import base, dfp, clean


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


class TestDfpFunction:
    """Test suite for the dfp (interpolate) preprocessing function."""

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

    def test_dfp_interpolates_nulls_correctly(self, data_with_nulls):
        """Test that dfp function interpolates null values correctly within each symbol group."""
        result = dfp(data_with_nulls)

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

    def test_dfp_preserves_non_null_values(self, data_with_nulls):
        """Test that dfp function preserves existing non-null values."""
        result = dfp(data_with_nulls)

        original_non_nulls = data_with_nulls.drop_nulls()
        result_subset = result.select(original_non_nulls.columns)

        for row_idx in range(len(data_with_nulls)):
            for col in data_with_nulls.columns:
                original_val = data_with_nulls[col][row_idx]
                if original_val is not None:
                    assert result[col][row_idx] == original_val

    def test_dfp_handles_data_without_nulls(self, data_without_nulls):
        """Test that dfp function handles data without nulls correctly."""
        result = dfp(data_without_nulls)

        assert result.equals(data_without_nulls)

    def test_dfp_handles_empty_dataframe(self, empty_dataframe):
        """Test that dfp function handles empty DataFrame correctly."""
        result = dfp(empty_dataframe)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert result.columns == empty_dataframe.columns

    def test_dfp_handles_single_row(self, single_row_data):
        """Test that dfp function handles single row DataFrame."""
        result = dfp(single_row_data)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        assert result["open"][0] is None

    def test_dfp_interpolates_per_symbol_group(self, data_with_nulls):
        """Test that interpolation is performed separately for each symbol."""
        result = dfp(data_with_nulls)

        aapl_result = result.filter(pl.col("symbol") == "AAPL")
        googl_result = result.filter(pl.col("symbol") == "GOOGL")

        assert len(aapl_result) == 4
        assert len(googl_result) == 3

        aapl_opens = aapl_result["open"].to_list()
        googl_opens = googl_result["open"].to_list()

        assert 150.0 in aapl_opens and 153.0 in aapl_opens
        assert 100.0 in googl_opens and 102.0 in googl_opens

    def test_dfp_handles_all_nulls_in_column_per_symbol(self, data_all_nulls_per_symbol):
        """Test dfp behavior when all values in a column are null for a symbol."""
        result = dfp(data_all_nulls_per_symbol)

        aapl_result = result.filter(pl.col("symbol") == "AAPL")
        googl_result = result.filter(pl.col("symbol") == "GOOGL")

        aapl_opens = aapl_result["open"].to_list()
        googl_closes = googl_result["close"].to_list()

        assert all(val is None for val in aapl_opens)
        assert all(val is None for val in googl_closes)

    def test_dfp_preserves_all_columns(self, data_with_nulls):
        """Test that dfp function preserves all columns from input."""
        result = dfp(data_with_nulls)

        assert result.columns == data_with_nulls.columns
        assert len(result) == len(data_with_nulls)

    def test_dfp_returns_new_dataframe(self, data_with_nulls):
        """Test that dfp function returns a new DataFrame object."""
        result = dfp(data_with_nulls)

        assert result is not data_with_nulls
        assert isinstance(result, pl.DataFrame)

    def test_dfp_handles_edge_nulls(self, data_first_last_nulls):
        """Test dfp behavior with nulls at the beginning and end of sequences."""
        result = dfp(data_first_last_nulls)

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

    def test_dfp_with_mixed_data_types(self):
        """Test dfp function with various data types including some that shouldn't be interpolated."""
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

        result = dfp(mixed_data)

        assert result["open"][1] == 151.0
        assert result["close"][1] == 155.0
        assert result["volume"][1] == 1100000

    def test_dfp_multiple_symbols_independence(self):
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

        result = dfp(multi_symbol_data)

        aapl_price = result.filter(pl.col("symbol") == "AAPL")["price"].to_list()
        googl_price = result.filter(pl.col("symbol") == "GOOGL")["price"].to_list()
        msft_price = result.filter(pl.col("symbol") == "MSFT")["price"].to_list()

        # Since each symbol only has 2 values with the second being null,
        # interpolation cannot happen (boundary nulls remain null)
        assert aapl_price[1] is None
        assert googl_price[1] is None
        assert msft_price[1] is None


class TestIntegrationBaseAndDfp:
    """Integration tests for base and dfp functions used together."""

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

    def test_base_then_dfp_integration(self, unsorted_data_with_nulls):
        """Test using base and dfp functions in sequence."""
        sorted_data = base(unsorted_data_with_nulls)
        interpolated_data = dfp(sorted_data)

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
        first_pass = dfp(base(unsorted_data_with_nulls))
        second_pass = dfp(base(first_pass))

        assert first_pass.equals(second_pass)


class TestCleanFunction:
    """Test suite for the clean function."""

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

    def test_clean_removes_boundary_nulls(self, data_with_boundary_nulls):
        """Test that clean function removes rows with boundary nulls."""
        result = clean(data_with_boundary_nulls)

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

    def test_clean_preserves_data_without_nulls(self, data_no_nulls):
        """Test that clean function preserves data when there are no nulls."""
        result = clean(data_no_nulls)

        assert result.equals(data_no_nulls)
        assert len(result) == 3

    def test_clean_handles_all_nulls_data(self, data_all_nulls):
        """Test that clean function handles data where all rows have nulls."""
        result = clean(data_all_nulls)

        # Should remove all rows since every row has at least one null
        assert len(result) == 0
        assert result.columns == data_all_nulls.columns

    def test_clean_handles_mixed_nulls(self, data_mixed_nulls):
        """Test that clean function handles scattered nulls correctly."""
        result = clean(data_mixed_nulls)

        # Should keep only rows with no nulls at all
        expected_rows = 2  # Only rows at index 0 and 3 have no nulls
        assert len(result) == expected_rows

        # Verify the remaining rows
        expected_prices = [100.0, 103.0]
        expected_volumes = [1000, 1300]
        assert result["price"].to_list() == expected_prices
        assert result["volume"].to_list() == expected_volumes

    def test_clean_handles_empty_dataframe(self, empty_dataframe):
        """Test that clean function handles empty DataFrame."""
        result = clean(empty_dataframe)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert result.columns == empty_dataframe.columns

    def test_clean_preserves_column_types(self, data_with_boundary_nulls):
        """Test that clean function preserves column data types."""
        result = clean(data_with_boundary_nulls)

        original_schema = data_with_boundary_nulls.schema
        result_schema = result.schema

        assert result_schema == original_schema

    def test_clean_preserves_column_order(self, data_with_boundary_nulls):
        """Test that clean function preserves column order."""
        result = clean(data_with_boundary_nulls)

        assert result.columns == data_with_boundary_nulls.columns

    def test_clean_returns_new_dataframe(self, data_no_nulls):
        """Test that clean function returns a new DataFrame object."""
        result = clean(data_no_nulls)

        assert result is not data_no_nulls
        assert isinstance(result, pl.DataFrame)

    def test_clean_maintains_sorting_order(self):
        """Test that clean function maintains the sorted order of data."""
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

        result = clean(sorted_data_with_nulls)

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

    def test_clean_handles_single_row_with_null(self):
        """Test clean function with single row containing null."""
        single_row_null = pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [date(2023, 1, 1)],
                "price": [None],
            }
        )

        result = clean(single_row_null)

        assert len(result) == 0
        assert result.columns == single_row_null.columns

    def test_clean_handles_single_row_without_null(self):
        """Test clean function with single row without null."""
        single_row_no_null = pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [date(2023, 1, 1)],
                "price": [100.0],
            }
        )

        result = clean(single_row_no_null)

        assert result.equals(single_row_no_null)
        assert len(result) == 1

    def test_clean_with_multiple_symbol_groups(self):
        """Test that clean function works correctly across multiple symbol groups."""
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

        result = clean(multi_symbol_data)

        # Should keep only the complete rows
        assert len(result) == 2
        expected_symbols = ["AAPL", "GOOGL"]
        expected_prices = [100.0, 200.0]

        assert result["symbol"].to_list() == expected_symbols
        assert result["price"].to_list() == expected_prices


class TestIntegrationAllFunctions:
    """Integration tests for base, dfp, and clean functions used together."""

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
        """Test the complete preprocessing pipeline: base -> dfp -> clean."""
        # Step 1: Sort the data
        sorted_data = base(raw_unsorted_data_with_gaps)

        # Step 2: Interpolate missing values
        interpolated_data = dfp(sorted_data)

        # Step 3: Clean boundary nulls
        cleaned_data = clean(interpolated_data)

        # Verify the pipeline worked correctly
        assert len(cleaned_data) > 0  # Should have some data remaining
        assert cleaned_data.null_count().sum_horizontal()[0] == 0  # No nulls should remain

        # Verify data is still sorted
        symbols = cleaned_data["symbol"].to_list()
        dates = cleaned_data["date"].to_list()

        # Check that symbols are in order and dates within symbols are ascending
        for i in range(len(symbols) - 1):
            if symbols[i] == symbols[i + 1]:
                assert dates[i] <= dates[i + 1]
            else:
                assert symbols[i] <= symbols[i + 1]

    def test_pipeline_functions_are_composable(self, raw_unsorted_data_with_gaps):
        """Test that the functions can be composed together."""
        # Single composition
        result = clean(dfp(base(raw_unsorted_data_with_gaps)))

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

        result = clean(dfp(base(perfect_data)))

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

        result = clean(dfp(base(edge_case_data)))

        # Should only keep AAPL data
        assert len(result) == 2
        assert all(symbol == "AAPL" for symbol in result["symbol"].to_list())

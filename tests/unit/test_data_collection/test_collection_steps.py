"""Unit tests for data collection step functions."""

import pandas as pd
import polars as pl
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import date, datetime
from stock_market_analytics.config import config

from stock_market_analytics.data_collection.collection_steps import (
    load_tickers,
    load_metadata,
    create_collection_plan,
    create_collection_plans,
    collect_and_process_symbol,
    update_metadata,
    update_historical_data,
    _initialize_metadata_record,
    _collect_raw_data,
    _process_and_validate_data,
    _combine_with_existing_data,
    _clean_and_deduplicate,
)


class TestLoadTickers:
    """Test suite for load_tickers function."""

    def test_load_tickers_success(self, tmp_path):
        """Test successful ticker loading."""
        tickers_file = tmp_path / config.data_collection.tickers_file
        tickers_data = """Symbol,Name,Country,IPO Year,Sector,Industry
AAPL,Apple Inc.,USA,1980,Technology,Consumer Electronics
GOOGL,Alphabet Inc.,USA,2004,Technology,Internet Content & Information"""
        tickers_file.write_text(tickers_data)

        result = load_tickers(str(tickers_file))

        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[1]["symbol"] == "GOOGL"
        assert "name" in result[0]
        assert "sector" in result[0]

    def test_load_tickers_file_not_found(self, tmp_path):
        """Test FileNotFoundError when tickers file doesn't exist."""
        with pytest.raises(ValueError, match="Error loading tickers file"):
            load_tickers(str(tmp_path / config.data_collection.tickers_file))

    def test_load_tickers_invalid_format(self, tmp_path):
        """Test ValueError when file format is invalid."""
        tickers_file = tmp_path / config.data_collection.tickers_file
        tickers_file.write_text("invalid,format")

        with pytest.raises(ValueError, match="Error loading tickers file"):
            load_tickers(str(tickers_file))


class TestLoadMetadata:
    """Test suite for load_metadata function."""

    def test_load_metadata_success(self, tmp_path):
        """Test successful metadata loading."""
        metadata_file = tmp_path / "metadata.csv"
        metadata_data = """symbol,last_ingestion,max_date_recorded,status
AAPL,2023-01-01,2023-01-01,active
GOOGL,2023-01-02,2023-01-02,active"""
        metadata_file.write_text(metadata_data)

        result = load_metadata(str(metadata_file))

        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["status"] == "active"

    def test_load_metadata_file_not_found(self, tmp_path):
        """Test empty list when metadata file doesn't exist."""
        result = load_metadata(str(tmp_path / "metadata.csv"))
        assert result == []

    def test_load_metadata_empty_file(self, tmp_path):
        """Test empty list when metadata file is empty."""
        metadata_file = tmp_path / "metadata.csv"
        metadata_file.write_text("symbol,last_ingestion,max_date_recorded,status\n")

        result = load_metadata(str(metadata_file))
        assert result == []

    def test_load_metadata_missing_columns(self, tmp_path):
        """Test ValueError when required columns are missing."""
        metadata_file = tmp_path / "metadata.csv"
        metadata_file.write_text("symbol,status\nAAPL,active")

        with pytest.raises(ValueError, match="Metadata file is missing columns"):
            load_metadata(str(metadata_file))


class TestCreateCollectionPlan:
    """Test suite for create_collection_plan function."""

    def test_create_collection_plan_new_symbol(self):
        """Test collection plan for new symbol."""
        result = create_collection_plan("AAPL", None)

        assert result == {"symbol": "AAPL", "period": "max"}

    def test_create_collection_plan_inactive_symbol(self):
        """Test collection plan for inactive symbol."""
        metadata = {"symbol": "AAPL", "status": "inactive"}
        result = create_collection_plan("AAPL", metadata)

        assert result is None

    def test_create_collection_plan_active_symbol(self):
        """Test collection plan for active symbol with incremental update."""
        metadata = {
            "symbol": "AAPL",
            "status": "active",
            "last_ingestion": pd.Timestamp("2023-01-01"),
        }
        result = create_collection_plan("AAPL", metadata)

        assert result["symbol"] == "AAPL"
        assert "start" in result
        assert result["start"] == "2023-01-02"


class TestCreateCollectionPlans:
    """Test suite for create_collection_plans function."""

    def test_create_collection_plans_mixed(self):
        """Test creating collection plans for mixed ticker types."""
        tickers = [{"symbol": "AAPL"}, {"symbol": "GOOGL"}, {"symbol": "MSFT"}]
        metadata_info = [
            {
                "symbol": "AAPL",
                "status": "active",
                "last_ingestion": pd.Timestamp("2023-01-01"),
            },
            {
                "symbol": "GOOGL",
                "status": "inactive",
                "last_ingestion": pd.Timestamp("2023-01-01"),
            },
        ]

        result = create_collection_plans(tickers, metadata_info)

        # Should have plans for AAPL (active) and MSFT (new), but not GOOGL (inactive)
        assert len(result) == 2
        symbols = [plan["symbol"] for plan in result]
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" not in symbols


class TestInitializeMetadataRecord:
    """Test suite for _initialize_metadata_record function."""

    def test_initialize_metadata_record(self):
        """Test metadata record initialization."""
        result = _initialize_metadata_record("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["status"] == "collection_issue"
        assert result["max_date_recorded"] is None
        assert "last_ingestion" in result


class TestCollectRawData:
    """Test suite for _collect_raw_data function."""

    @patch("stock_market_analytics.data_collection.collection_steps.YFinanceCollector")
    def test_collect_raw_data_success(self, mock_collector_class):
        """Test successful raw data collection."""
        # Setup mock collector
        mock_collector = Mock()
        mock_collector.collection_successful = True
        mock_data = pl.DataFrame({"date": ["2023-01-01"], "close": [100.0]})
        mock_collector.get_historical_data.return_value = mock_data
        mock_collector_class.return_value = mock_collector

        collection_plan = {"symbol": "AAPL", "period": "max"}
        data, success = _collect_raw_data(collection_plan)

        assert success is True
        assert data is not None
        mock_collector_class.assert_called_once_with(**collection_plan)

    @patch("stock_market_analytics.data_collection.collection_steps.YFinanceCollector")
    def test_collect_raw_data_failure(self, mock_collector_class):
        """Test failed raw data collection."""
        # Setup mock collector
        mock_collector = Mock()
        mock_collector.collection_successful = False
        mock_collector.get_historical_data.return_value = pl.DataFrame()
        mock_collector_class.return_value = mock_collector

        collection_plan = {"symbol": "AAPL", "period": "max"}
        data, success = _collect_raw_data(collection_plan)

        assert success is False
        assert data is None


class TestProcessAndValidateData:
    """Test suite for _process_and_validate_data function."""

    @patch(
        "stock_market_analytics.data_collection.collection_steps.DataQualityValidator"
    )
    @patch(
        "stock_market_analytics.data_collection.collection_steps.ContinuousTimelineProcessor"
    )
    def test_process_and_validate_data_success(
        self, mock_processor_class, mock_validator_class
    ):
        """Test successful data processing and validation."""
        # Setup mock processor
        mock_processor = Mock()
        mock_processor.processing_successful = True
        processed_data = pl.DataFrame({"date": ["2023-01-01"], "close": [100.0]})
        mock_processor.process.return_value = processed_data
        mock_processor_class.return_value = mock_processor

        # Setup mock validator
        mock_validator = Mock()
        mock_validator.validation_successful = True
        validated_data = pl.DataFrame({"date": ["2023-01-01"], "close": [100.0]})
        mock_validator.validate.return_value = validated_data
        mock_validator_class.return_value = mock_validator

        raw_data = pl.DataFrame({"date": ["2023-01-01"], "close": [100.0]})
        result_data, status = _process_and_validate_data("AAPL", raw_data)

        assert status == "active"
        assert result_data is not None

    @patch(
        "stock_market_analytics.data_collection.collection_steps.ContinuousTimelineProcessor"
    )
    def test_process_and_validate_data_processing_failure(self, mock_processor_class):
        """Test data processing failure."""
        # Setup mock processor
        mock_processor = Mock()
        mock_processor.processing_successful = False
        mock_processor.process.return_value = None
        mock_processor_class.return_value = mock_processor

        raw_data = pl.DataFrame({"date": ["2023-01-01"], "close": [100.0]})
        result_data, status = _process_and_validate_data("AAPL", raw_data)

        assert status == "data_issue"
        assert result_data is None

    @patch(
        "stock_market_analytics.data_collection.collection_steps.DataQualityValidator"
    )
    @patch(
        "stock_market_analytics.data_collection.collection_steps.ContinuousTimelineProcessor"
    )
    def test_process_and_validate_data_validation_failure(
        self, mock_processor_class, mock_validator_class
    ):
        """Test data validation failure."""
        # Setup mock processor (success)
        mock_processor = Mock()
        mock_processor.processing_successful = True
        processed_data = pl.DataFrame({"date": ["2023-01-01"], "close": [100.0]})
        mock_processor.process.return_value = processed_data
        mock_processor_class.return_value = mock_processor

        # Setup mock validator (failure)
        mock_validator = Mock()
        mock_validator.validation_successful = False
        mock_validator.validate.return_value = None
        mock_validator_class.return_value = mock_validator

        raw_data = pl.DataFrame({"date": ["2023-01-01"], "close": [100.0]})
        result_data, status = _process_and_validate_data("AAPL", raw_data)

        assert status == "data_quality_issue"
        assert result_data is None


class TestCollectAndProcessSymbol:
    """Test suite for collect_and_process_symbol function."""

    @patch(
        "stock_market_analytics.data_collection.collection_steps._process_and_validate_data"
    )
    @patch("stock_market_analytics.data_collection.collection_steps._collect_raw_data")
    def test_collect_and_process_symbol_success(self, mock_collect, mock_process):
        """Test successful symbol collection and processing."""
        # Setup mocks
        mock_data = pl.DataFrame({"date": [date(2023, 1, 1)], "close": [100.0]})
        mock_collect.return_value = (mock_data, True)
        mock_process.return_value = (mock_data, "active")

        collection_plan = {"symbol": "AAPL", "period": "max"}
        result = collect_and_process_symbol(collection_plan)

        assert result["data"] is not None
        assert result["new_metadata"]["symbol"] == "AAPL"
        assert result["new_metadata"]["status"] == "active"
        assert "2023-01-01" in result["new_metadata"]["max_date_recorded"]

    @patch("stock_market_analytics.data_collection.collection_steps._collect_raw_data")
    def test_collect_and_process_symbol_collection_failure(self, mock_collect):
        """Test symbol collection failure."""
        mock_collect.return_value = (None, False)

        collection_plan = {"symbol": "AAPL", "period": "max"}
        result = collect_and_process_symbol(collection_plan)

        assert result["data"] is None
        assert result["new_metadata"]["status"] == "collection_issue"

    @patch("stock_market_analytics.data_collection.collection_steps._collect_raw_data")
    def test_collect_and_process_symbol_exception(self, mock_collect):
        """Test exception handling during symbol collection."""
        mock_collect.side_effect = Exception("Network error")

        collection_plan = {"symbol": "AAPL", "period": "max"}
        result = collect_and_process_symbol(collection_plan)

        assert result["data"] is None
        assert result["new_metadata"]["status"] == "collection_error"


class TestUpdateMetadata:
    """Test suite for update_metadata function."""

    def test_update_metadata_no_updates(self, tmp_path):
        """Test update metadata with no updates."""
        update_metadata(str(tmp_path / "metadata.csv"), [])
        # Should not create any files
        assert not (tmp_path / "metadata.csv").exists()

    def test_update_metadata_new_file(self, tmp_path):
        """Test update metadata creating new file."""
        metadata_updates = [
            {"symbol": "AAPL", "status": "active", "last_ingestion": "2023-01-01"}
        ]
        metadata_file = tmp_path / "metadata.csv"

        update_metadata(str(metadata_file), metadata_updates)
        assert metadata_file.exists()

        df = pd.read_csv(metadata_file)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"

    def test_update_metadata_existing_file(self, tmp_path):
        """Test update metadata with existing file."""
        # Create existing metadata file
        existing_file = tmp_path / "metadata.csv"
        existing_data = "symbol,status,last_ingestion\nGOOGL,active,2023-01-01\n"
        existing_file.write_text(existing_data)

        metadata_updates = [
            {"symbol": "AAPL", "status": "active", "last_ingestion": "2023-01-02"}
        ]

        update_metadata(str(existing_file), metadata_updates)

        df = pd.read_csv(existing_file)
        assert len(df) == 2
        symbols = df["symbol"].tolist()
        assert "GOOGL" in symbols
        assert "AAPL" in symbols


class TestCombineWithExistingData:
    """Test suite for _combine_with_existing_data function."""

    def test_combine_with_existing_data_no_existing(self, tmp_path):
        """Test combining data when no existing file."""
        stocks_history_path = tmp_path / "stocks_history.parquet"
        new_data = pl.DataFrame(
            {
                "date": [date(2023, 1, 1)],
                "symbol": ["AAPL"],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [102.0],
                "volume": [1000000],
            }
        )

        result = _combine_with_existing_data(str(stocks_history_path), new_data)

        assert result.equals(new_data)

    def test_combine_with_existing_data_with_existing(self, tmp_path):
        """Test combining data with existing file."""
        stocks_history_path = tmp_path / "stocks_history.parquet"

        # Create existing data
        existing_data = pl.DataFrame(
            {
                "date": [date(2023, 1, 1)],
                "symbol": ["GOOGL"],
                "open": [200.0],
                "high": [205.0],
                "low": [199.0],
                "close": [202.0],
                "volume": [2000000],
            }
        )
        existing_data.write_parquet(stocks_history_path)

        new_data = pl.DataFrame(
            {
                "date": [date(2023, 1, 2)],
                "symbol": ["AAPL"],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [102.0],
                "volume": [1000000],
            }
        )

        result = _combine_with_existing_data(str(stocks_history_path), new_data)

        assert len(result) == 2
        symbols = result["symbol"].to_list()
        assert "GOOGL" in symbols
        assert "AAPL" in symbols


class TestCleanAndDeduplicate:
    """Test suite for _clean_and_deduplicate function."""

    def test_clean_and_deduplicate_basic(self):
        """Test basic cleaning and deduplication."""
        data = pl.DataFrame(
            {
                "date": [date(2023, 1, 2), date(2023, 1, 1), date(2023, 1, 1)],
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "open": [101.0, 100.0, 99.0],
                "high": [106.0, 105.0, 104.0],
                "low": [98.0, 99.0, 97.0],
                "close": [103.0, 102.0, 101.0],
                "volume": [1100000, 1000000, 900000],
            }
        )

        result = _clean_and_deduplicate(data)

        # Should be sorted by symbol, date and deduplicated
        assert len(result) == 2  # Duplicates removed
        assert result["date"].to_list() == [date(2023, 1, 1), date(2023, 1, 2)]
        # Should keep the last entry for duplicates (close=101.0)
        assert result.filter(pl.col("date") == date(2023, 1, 1))["close"][0] == 101.0


class TestUpdateHistoricalData:
    """Test suite for update_historical_data function."""

    def test_update_historical_data_no_data(self, tmp_path):
        """Test update with no collected data."""
        result = update_historical_data(str(tmp_path / "stocks_history.parquet"), [])

        assert result["status"] == "no_new_data"

    def test_update_historical_data_success(self, tmp_path):
        """Test successful historical data update."""
        collected_data = [
            pl.DataFrame(
                {
                    "date": [date(2023, 1, 1)],
                    "symbol": ["AAPL"],
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [102.0],
                    "volume": [1000000],
                }
            )
        ]

        result = update_historical_data(
            str(tmp_path / "stocks_history.parquet"), collected_data
        )

        assert result["status"] == "success"
        assert result["total_records"] == 1
        assert result["new_records"] == 1
        assert result["symbols_updated"] == 1

        # Verify file was created
        stocks_file = tmp_path / "stocks_history.parquet"
        assert stocks_file.exists()

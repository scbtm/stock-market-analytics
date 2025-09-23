"""
Coordinate core data collection components and
reuse across different flows and scenarios.
"""

from typing import Any

import pandas as pd
import polars as pl

from stock_market_analytics.config import config
from stock_market_analytics.data_collection import (
    ContinuousTimelineProcessor,
    DataQualityValidator,
    YFinanceCollector,
)


def load_tickers(tickers_path: str) -> list[dict[str, Any]]:
    """Load and validate ticker symbols from CSV file."""
    required_columns = config.data_collection.required_ticker_columns
    column_mapping = config.data_collection.ticker_column_mapping

    try:
        tickers_df = pd.read_csv(tickers_path, usecols=required_columns)
        tickers_df = tickers_df.rename(columns=column_mapping)
        return tickers_df.to_dict(orient="records")  # type: ignore

    except Exception as e:
        raise ValueError(f"Error loading tickers file: {str(e)}") from e


def load_metadata(metadata_path: str) -> list[dict[str, Any]]:
    """Load and validate existing metadata from CSV file."""
    required_columns = config.data_collection.required_metadata_columns

    try:
        metadata_df = pd.read_csv(metadata_path)
        if metadata_df.empty:
            return []

    except Exception:
        return []

    try:
        # Validate required columns
        missing_cols = set(required_columns) - set(metadata_df.columns)
        if missing_cols:
            raise ValueError(f"Metadata file is missing columns: {missing_cols}")

        # Ensure proper data types
        metadata_df["last_ingestion"] = pd.to_datetime(
            metadata_df["last_ingestion"], errors="coerce"
        )
        metadata_df["max_date_recorded"] = pd.to_datetime(
            metadata_df["max_date_recorded"], errors="coerce"
        )

        return metadata_df.to_dict(orient="records")  # type: ignore

    except Exception as e:
        raise ValueError(f"Error loading metadata file: {str(e)}") from e


def create_collection_plan(
    symbol: str, metadata: dict[str, Any] | None
) -> dict[str, str] | None:
    """Create a collection plan for a single symbol."""
    if not metadata:
        # New symbol - collect full history
        return {"symbol": symbol, "period": "max"}

    # Check if symbol is active and worth collecting
    if metadata["status"] != "active":
        return None

    # Existing active symbol - collect incremental updates
    last_ingestion = metadata["last_ingestion"]
    start_date = (last_ingestion + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    return {"symbol": symbol, "start": start_date}


def create_collection_plans(
    tickers: list[dict[str, Any]], metadata_info: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Create collection plans for all ticker symbols."""
    collection_plans = []
    metadata_lookup = {item["symbol"]: item for item in metadata_info}

    for ticker in tickers:
        symbol = ticker["symbol"]
        plan = create_collection_plan(symbol, metadata_lookup.get(symbol))

        if plan:  # Only add valid plans
            collection_plans.append(plan)

    return collection_plans


def _initialize_metadata_record(symbol: str) -> dict[str, Any]:
    """Initialize a new metadata record for a symbol."""
    return {
        "symbol": symbol,
        "last_ingestion": pd.Timestamp.now().normalize().date().strftime("%Y-%m-%d"),
        "max_date_recorded": None,
        "status": "collection_issue",
    }


def _collect_raw_data(
    collection_plan: dict[str, str],
) -> tuple[pl.DataFrame | None, bool]:
    """Collect raw data for a symbol."""
    collector = YFinanceCollector(**collection_plan)
    raw_data = collector.get_historical_data()
    success = collector.collection_successful and not raw_data.is_empty()
    return raw_data if success else None, success


def _process_and_validate_data(
    symbol: str, raw_data: pl.DataFrame
) -> tuple[pl.DataFrame | None, str]:
    """Process and validate collected data."""
    # Process for continuity
    processor = ContinuousTimelineProcessor(symbol, raw_data)
    processed_data = processor.process()

    if processed_data is None or not processor.processing_successful:
        return None, "data_issue"

    # Validate quality
    quality_validator = DataQualityValidator(symbol, processed_data)
    validated_data = quality_validator.validate()

    if validated_data is None or not quality_validator.validation_successful:
        return None, "data_quality_issue"

    return validated_data, "active"


def collect_and_process_symbol(collection_plan: dict[str, str]) -> dict[str, Any]:
    """Collect and process data for a single stock symbol."""
    symbol = collection_plan["symbol"]
    new_metadata = _initialize_metadata_record(symbol)

    try:
        # Collect raw data
        raw_data, collection_success = _collect_raw_data(collection_plan)
        if not collection_success or raw_data is None:
            return {"data": None, "new_metadata": new_metadata}

        # Process and validate
        validated_data, status = _process_and_validate_data(symbol, raw_data)
        new_metadata["status"] = status

        if validated_data is not None:
            new_metadata["max_date_recorded"] = (
                validated_data["date"].max().strftime("%Y-%m-%d")
            )

        return {"data": validated_data, "new_metadata": new_metadata}

    except Exception:
        new_metadata["status"] = "collection_error"
        return {"data": None, "new_metadata": new_metadata}


def update_metadata(metadata_path: str, metadata_updates: list[dict[str, Any]]) -> None:
    """Update the metadata file with new collection information."""
    if not metadata_updates:
        return

    new_metadata_df = pd.DataFrame(metadata_updates)

    try:
        existing_metadata_df = pd.read_csv(metadata_path)
        if not existing_metadata_df.empty:
            combined_df = pd.concat([existing_metadata_df, new_metadata_df])
        else:
            combined_df = new_metadata_df

    except Exception:
        combined_df = new_metadata_df

    # Clean up metadata: keep only the latest entry per symbol
    final_metadata_df = (
        combined_df.sort_values(["symbol", "last_ingestion"], ascending=[True, True])
        .drop_duplicates(subset=["symbol"], keep="last")
        .reset_index(drop=True)
    )

    # Save updated metadata
    final_metadata_df.to_csv(metadata_path, index=False)


def _combine_with_existing_data(
    stocks_history_path: str, new_data: pl.DataFrame
) -> pl.DataFrame:
    """Combine new data with existing historical data."""

    try:
        existing_data = pl.read_parquet(stocks_history_path)
        existing_data = existing_data.select(
            ["date", "symbol", "open", "high", "low", "close", "volume"]
        ).cast({"volume": pl.Int64})
        return pl.concat([existing_data, new_data])
    except Exception:
        return new_data


def _clean_and_deduplicate(data: pl.DataFrame) -> pl.DataFrame:
    """Clean and deduplicate historical data."""
    return (
        data.sort(["symbol", "date"], descending=[False, False])
        .group_by(["symbol", "date"])
        .agg(pl.all().last())
        .sort(["symbol", "date"], descending=[False, False])
        .select(["date", "symbol", "open", "high", "low", "close", "volume"])
    )


def update_historical_data(
    stocks_history_path: str, collected_data: list[pl.DataFrame]
) -> dict[str, Any]:
    """Update the historical data parquet file with newly collected data."""
    if not collected_data:
        return {"status": "no_new_data"}

    new_data = pl.concat(collected_data)

    # Combine and clean data
    combined_data = _combine_with_existing_data(stocks_history_path, new_data)
    final_data = _clean_and_deduplicate(combined_data)

    # Save updated data
    final_data.write_parquet(stocks_history_path)

    return {
        "status": "success",
        "total_records": len(final_data),
        "new_records": len(new_data),
        "symbols_updated": new_data["symbol"].n_unique(),
    }

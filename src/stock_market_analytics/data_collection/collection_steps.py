"""
Coordinate core data collection components and 
reuse across different flows and scenarios.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from stock_market_analytics.config import config
from stock_market_analytics.data_collection import (
    ContinuousTimelineProcessor,
    DataQualityValidator, 
    YFinanceCollector,
)


def load_tickers(base_data_path: Path) -> list[dict[str, Any]]:
    """Load and validate ticker symbols from CSV file."""
    tickers_file = config.data_collection.tickers_file
    required_columns = config.data_collection.required_ticker_columns
    column_mapping = config.data_collection.ticker_column_mapping
    
    tickers_path = base_data_path / tickers_file

    if not tickers_path.exists():
        raise FileNotFoundError(
            f"Tickers file not found at {tickers_path}. "
            "A set of stocks to collect must be provided."
        )

    try:
        tickers_df = pd.read_csv(tickers_path, usecols=required_columns)
        tickers_df = tickers_df.rename(columns=column_mapping)
        return tickers_df.to_dict(orient="records")  # type: ignore

    except Exception as e:
        raise ValueError(f"Error loading tickers file: {str(e)}") from e


def load_metadata(base_data_path: Path) -> list[dict[str, Any]]:
    """Load and validate existing metadata from CSV file."""
    metadata_file = config.data_collection.metadata_file
    required_columns = config.data_collection.required_metadata_columns
    
    metadata_path = base_data_path / metadata_file

    if not metadata_path.exists():
        return []

    try:
        metadata_df = pd.read_csv(metadata_path)

        if metadata_df.empty:
            raise ValueError("Metadata file is empty")

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


def collect_and_process_symbol(collection_plan: dict[str, str]) -> dict[str, Any]:
    """Collect and process data for a single stock symbol."""
    symbol = collection_plan["symbol"]

    # Initialize metadata record
    new_metadata = {
        "symbol": symbol,
        "last_ingestion": pd.Timestamp.now().normalize().date().strftime("%Y-%m-%d"),
        "max_date_recorded": None,
        "status": "collection_issue",
    }

    try:
        # Collect raw data
        collector = YFinanceCollector(**collection_plan)
        raw_data = collector.get_historical_data()

        if not collector.collection_successful or raw_data.is_empty():
            return {"data": None, "new_metadata": new_metadata}

        # Process data for continuity and validation
        processor = ContinuousTimelineProcessor(symbol, raw_data)
        processed_data = processor.process()

        if processed_data is not None and processor.processing_successful:
            # Apply data quality validation
            quality_validator = DataQualityValidator(symbol, processed_data)
            validated_data = quality_validator.validate()

            if validated_data is not None and quality_validator.validation_successful:
                new_metadata["max_date_recorded"] = (
                    validated_data["date"].max().strftime("%Y-%m-%d")
                )
                new_metadata["status"] = "active"
                return {"data": validated_data, "new_metadata": new_metadata}
            else:
                # Quality check failed
                new_metadata["status"] = "data_quality_issue"
                return {"data": None, "new_metadata": new_metadata}
        else:
            new_metadata["status"] = "data_issue"
            return {"data": None, "new_metadata": new_metadata}

    except Exception:
        new_metadata["status"] = "collection_error"
        return {"data": None, "new_metadata": new_metadata}


def update_metadata(
    base_data_path: Path, metadata_updates: list[dict[str, Any]]
) -> None:
    """Update the metadata file with new collection information."""
    if not metadata_updates:
        return

    metadata_file = config.data_collection.metadata_file
    metadata_path = base_data_path / metadata_file
    new_metadata_df = pd.DataFrame(metadata_updates)

    if metadata_path.exists():
        existing_metadata_df = pd.read_csv(metadata_path)
        combined_df = pd.concat([existing_metadata_df, new_metadata_df])
    else:
        combined_df = new_metadata_df

    # Clean up metadata: keep only the latest entry per symbol
    final_metadata_df = (
        combined_df.sort_values(["symbol", "last_ingestion"], ascending=[True, True])
        .drop_duplicates(subset=["symbol"], keep="last")
        .reset_index(drop=True)
    )

    # Save updated metadata
    final_metadata_df.to_csv(metadata_path, index=False)


def update_historical_data(
    base_data_path: Path, collected_data: list[pl.DataFrame]
) -> dict[str, Any]:
    """Update the historical data parquet file with newly collected data."""
    if not collected_data:
        return {"status": "no_new_data"}

    stocks_history_file = config.data_collection.stocks_history_file
    stocks_history_path = base_data_path / stocks_history_file

    # Combine all new data
    new_data = pl.concat(collected_data)

    # Load existing data if available
    if stocks_history_path.exists():
        existing_data = pl.read_parquet(stocks_history_path)
        existing_data = existing_data.select(
            ["date", "symbol", "open", "high", "low", "close", "volume"]
        ).cast({"volume": pl.Int64})
        combined_data = pl.concat([existing_data, new_data])
    else:
        combined_data = new_data

    # Clean and deduplicate data
    final_data = (
        combined_data.sort(["symbol", "date"], descending=[False, False])
        .group_by(["symbol", "date"])
        .agg(pl.all().last())  # Keep the most recent entry for duplicates
        .sort(["symbol", "date"], descending=[False, False])
    )

    # Reorder columns to match storage format
    final_data = final_data.select(
        ["date", "symbol", "open", "high", "low", "close", "volume"]
    )

    # Save updated historical data
    final_data.write_parquet(stocks_history_path)

    return {
        "status": "success",
        "total_records": len(final_data),
        "new_records": len(new_data),
        "symbols_updated": new_data["symbol"].n_unique(),
    }
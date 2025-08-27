import os
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from metaflow import FlowSpec, step

from stock_market_analytics.data_collection import (
    ContinuousTimelineProcessor,
    YFinanceCollector,
)

# Constants
TICKERS_FILE = "tickers.csv"
METADATA_FILE = "metadata.csv"
STOCKS_HISTORY_FILE = "stocks_history.parquet"

# Required columns for tickers file
REQUIRED_TICKER_COLUMNS = [
    "Symbol",
    "Name",
    "Country",
    "IPO Year",
    "Sector",
    "Industry",
]
TICKER_COLUMN_MAPPING = {
    col: col.replace(" ", "_").lower() for col in REQUIRED_TICKER_COLUMNS
}

# Required columns for metadata file
REQUIRED_METADATA_COLUMNS = ["symbol", "last_ingestion", "max_date_recorded", "status"]


class BatchCollectionFlow(FlowSpec):
    """
    A Metaflow pipeline for batch collection and processing of stock market data.

    This flow orchestrates the entire data collection process:
    1. Loads ticker symbols and existing metadata
    2. Creates collection plans (full vs incremental updates)
    3. Collects data in parallel for each symbol
    4. Processes and validates the collected data
    5. Merges results and updates metadata

    The flow expects a BASE_DATA_PATH environment variable pointing to the data directory.
    """

    @step
    def start(self) -> None:
        """
        Initialize the batch collection flow.

        This is the entry point for the Metaflow pipeline. It validates the
        environment and begins the data collection process.
        """
        print("🚀 Starting Batch Collection Flow...")

        # Validate required environment variables
        if not os.environ.get("BASE_DATA_PATH"):
            raise ValueError("BASE_DATA_PATH environment variable is required")

        print(f"📁 Data directory: {os.environ['BASE_DATA_PATH']}")
        self.next(self.load_inputs)

    @step
    def load_inputs(self) -> None:
        """
        Load ticker symbols and existing metadata from the data directory.

        This step:
        1. Loads the ticker symbols from tickers.csv
        2. Validates the ticker file format
        3. Loads existing metadata (if available) for incremental updates
        4. Validates metadata file format

        Raises:
            FileNotFoundError: If tickers.csv is not found
            ValueError: If required columns are missing from files
        """
        print("📊 Loading ticker symbols and metadata...")

        base_data_path = Path(os.environ["BASE_DATA_PATH"])

        # Load and validate tickers
        self.tickers = self._load_tickers(base_data_path)
        print(f"✅ Loaded {len(self.tickers)} ticker symbols")

        # Load existing metadata if available
        self.metadata_info = self._load_metadata(base_data_path)
        if self.metadata_info:
            print(f"📋 Found existing metadata for {len(self.metadata_info)} symbols")
        else:
            print("📋 No existing metadata found - will perform full collection")

        self.next(self.build_collection_plan)

    def _load_tickers(self, base_data_path: Path) -> list[dict[str, Any]]:
        """Load and validate ticker symbols from CSV file."""
        tickers_path = base_data_path / TICKERS_FILE

        if not tickers_path.exists():
            raise FileNotFoundError(
                f"Tickers file not found at {tickers_path}. "
                "A set of stocks to collect must be provided."
            )

        try:
            tickers_df = pd.read_csv(tickers_path, usecols=REQUIRED_TICKER_COLUMNS)
            tickers_df = tickers_df.rename(columns=TICKER_COLUMN_MAPPING)

            return tickers_df.to_dict(orient="records")

        except Exception as e:
            raise ValueError(f"Error loading tickers file: {str(e)}") from e

    def _load_metadata(self, base_data_path: Path) -> list[dict[str, Any]]:
        """Load and validate existing metadata from CSV file."""
        metadata_path = base_data_path / METADATA_FILE

        if not metadata_path.exists():
            return []

        try:
            metadata_df = pd.read_csv(metadata_path)

            if metadata_df.empty:
                raise ValueError("Metadata file is empty")

            # Validate required columns
            missing_cols = set(REQUIRED_METADATA_COLUMNS) - set(metadata_df.columns)
            if missing_cols:
                raise ValueError(f"Metadata file is missing columns: {missing_cols}")

            # Ensure proper data types
            metadata_df["last_ingestion"] = pd.to_datetime(
                metadata_df["last_ingestion"], errors="coerce"
            )
            metadata_df["max_date_recorded"] = pd.to_datetime(
                metadata_df["max_date_recorded"], errors="coerce"
            )

            return metadata_df.to_dict(orient="records")

        except Exception as e:
            raise ValueError(f"Error loading metadata file: {str(e)}") from e

    @step
    def build_collection_plan(self) -> None:
        """
        Create collection plans for each ticker symbol.

        This step determines what data to collect for each symbol:
        - Full historical data ("max" period) for new symbols
        - Incremental updates (from last ingestion date) for existing symbols
        - Skips symbols with inactive status or recent collection failures

        The collection plans are then distributed to parallel collection tasks.
        """
        print("📅 Building collection plans...")

        collection_plans = []
        metadata_lookup = {item["symbol"]: item for item in self.metadata_info}

        for ticker in self.tickers:
            symbol = ticker["symbol"]
            plan = self._create_collection_plan(symbol, metadata_lookup.get(symbol))

            if plan:  # Only add valid plans
                collection_plans.append(plan)

        self.collection_plans = collection_plans

        print(f"🔄 Created {len(collection_plans)} collection plans")
        print("⚡ Starting parallel data collection...")

        # Distribute collection plans to parallel tasks
        self.next(self.collect_data, foreach="collection_plans")

    def _create_collection_plan(
        self, symbol: str, metadata: dict[str, Any] | None
    ) -> dict[str, str] | None:
        """
        Create a collection plan for a single symbol.

        Args:
            symbol: Stock symbol to collect
            metadata: Existing metadata for the symbol (if any)

        Returns:
            Collection plan dictionary or None if no collection needed
        """
        if not metadata:
            # New symbol - collect full history
            return {"symbol": symbol, "period": "max"}

        # Check if symbol is active and worth collecting
        if metadata["status"] != "active":
            print(f"⚠️ Skipping {symbol} - status: {metadata['status']}")
            return None

        # Existing active symbol - collect incremental updates
        last_ingestion = metadata["last_ingestion"]
        start_date = (last_ingestion + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        return {"symbol": symbol, "start": start_date}

    @step
    def collect_data(self) -> None:
        """
        Collect and process data for a single stock symbol.

        This step runs in parallel for each collection plan and:
        1. Uses YFinanceCollector to fetch raw data
        2. Processes data using ContinuousTimelineProcessor
        3. Creates metadata record with collection status
        4. Handles errors gracefully

        The results are passed to the join step for aggregation.
        """
        collection_plan = self.input
        symbol = collection_plan["symbol"]

        # Initialize metadata record
        new_metadata = {
            "symbol": symbol,
            "last_ingestion": pd.Timestamp.now()
            .normalize()
            .date()
            .strftime("%Y-%m-%d"),
            "max_date_recorded": None,
            "status": "collection_issue",
        }

        try:
            # Collect raw data
            collector = YFinanceCollector(**collection_plan)
            raw_data = collector.get_historical_data()

            if not collector.collection_successful or raw_data.is_empty():
                self.results = {"data": None, "new_metadata": new_metadata}
                self.next(self.join_results)
                return

            # Process data for continuity and validation
            processor = ContinuousTimelineProcessor(symbol, raw_data)
            processed_data = processor.process()

            if processed_data is not None and processor.processing_successful:
                new_metadata["max_date_recorded"] = (
                    processed_data["date"].max().strftime("%Y-%m-%d")
                )
                new_metadata["status"] = "active"
                self.results = {"data": processed_data, "new_metadata": new_metadata}
            else:
                new_metadata["status"] = "data_issue"
                self.results = {"data": None, "new_metadata": new_metadata}

        except Exception:
            new_metadata["status"] = "collection_error"
            self.results = {"data": None, "new_metadata": new_metadata}

        self.next(self.join_results)

    @step
    def join_results(self, inputs: list[dict[str, Any]]) -> None:
        """
        Aggregate results from all parallel collection tasks.

        This step:
        1. Collects all data and metadata from parallel tasks
        2. Merges new data with existing historical data
        3. Updates the metadata file with collection status
        4. Ensures data consistency and removes duplicates
        5. Persists results to parquet and CSV files

        Args:
            inputs: List of results from parallel collect_data steps
        """
        print("🔄 Joining results from all collection tasks...")

        results = [input_item.results for input_item in inputs]
        base_data_path = Path(os.environ["BASE_DATA_PATH"])

        # Separate successful data collections from metadata updates
        collected_data = []
        metadata_updates = []

        for result in results:
            if result["data"] is not None and not result["data"].is_empty():
                collected_data.append(result["data"])

            if result["new_metadata"] is not None:
                metadata_updates.append(result["new_metadata"])

        print(f"📊 Collected data for {len(collected_data)} symbols")
        print(f"📋 Processing {len(metadata_updates)} metadata updates")

        # Update metadata
        self._update_metadata(base_data_path, metadata_updates)

        # Update historical data
        self._update_historical_data(base_data_path, collected_data)

        print("✅ Successfully completed batch collection")
        self.next(self.end)

    def _update_metadata(
        self, base_data_path: Path, metadata_updates: list[dict[str, Any]]
    ) -> None:
        """Update the metadata file with new collection information."""
        if not metadata_updates:
            print("⚠️ No metadata updates to process")
            return

        metadata_path = base_data_path / METADATA_FILE
        new_metadata_df = pd.DataFrame(metadata_updates)

        if metadata_path.exists():
            existing_metadata_df = pd.read_csv(metadata_path)
            # Combine and deduplicate, keeping the most recent entry per symbol
            combined_df = pd.concat([existing_metadata_df, new_metadata_df])
        else:
            combined_df = new_metadata_df

        # Clean up metadata: keep only the latest entry per symbol
        final_metadata_df = (
            combined_df.sort_values(
                ["symbol", "last_ingestion"], ascending=[True, True]
            )
            .drop_duplicates(subset=["symbol"], keep="last")
            .reset_index(drop=True)
        )

        # Save updated metadata
        final_metadata_df.to_csv(metadata_path, index=False)
        print(f"📋 Updated metadata for {len(final_metadata_df)} symbols")

    def _update_historical_data(
        self, base_data_path: Path, collected_data: list[pl.DataFrame]
    ) -> None:
        """Update the historical data parquet file with newly collected data."""
        if not collected_data:
            print("⚠️ No new data to add to historical dataset")
            return

        stocks_history_path = base_data_path / STOCKS_HISTORY_FILE

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
            combined_data.sort(
                ["symbol", "date"], descending=[False, False]
            )  # Sort ascending for better performance
            .group_by(["symbol", "date"])
            .agg(pl.all().last())  # Keep the most recent entry for duplicates
            .sort(["symbol", "date"], descending=[False, False])
        )

        # Reorder columns to match storage format (symbol first, then date)
        final_data = final_data.select(
            ["date", "symbol", "open", "high", "low", "close", "volume"]
        )

        # Save updated historical data
        final_data.write_parquet(stocks_history_path)

        total_records = len(final_data)
        new_records = len(new_data)
        symbols_updated = new_data["symbol"].n_unique()

        print("📈 Updated historical dataset:")
        print(f"  • Added {new_records:,} new records")
        print(f"  • Updated {symbols_updated} symbols")
        print(f"  • Total records: {total_records:,}")

    @step
    def end(self) -> None:
        """
        Final step of the batch collection flow.

        This step marks the completion of the entire data collection pipeline.
        All data has been collected, processed, and persisted to the data directory.
        """
        print("✅ Batch Collection Flow completed successfully!")
        print("🎉 All stock data has been collected and updated.")


if __name__ == "__main__":
    # Entry point for running the flow directly
    BatchCollectionFlow()

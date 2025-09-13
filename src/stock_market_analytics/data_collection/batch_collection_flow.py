import os
from pathlib import Path
from typing import Any

from metaflow import FlowSpec, step
import wandb
from wandb.integration.metaflow import wandb_log

import pandas as pd

from stock_market_analytics.data_collection import collection_steps
from stock_market_analytics.config import config

# Initialize wandb
wandb.login(key=config.wandb_key)


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
        self.tickers = collection_steps.load_tickers(base_data_path)
        print(f"✅ Loaded {len(self.tickers)} ticker symbols")

        # Load existing metadata if available
        self.metadata_info = collection_steps.load_metadata(base_data_path)
        if self.metadata_info:
            print(f"📋 Found existing metadata for {len(self.metadata_info)} symbols")
        else:
            print("📋 No existing metadata found - will perform full collection")

        self.next(self.build_collection_plan)

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

        self.collection_plans = collection_steps.create_collection_plans(
            self.tickers, self.metadata_info
        )

        print(f"🔄 Created {len(self.collection_plans)} collection plans")
        print("⚡ Starting parallel data collection...")

        # Distribute collection plans to parallel tasks
        self.next(self.collect_data, foreach="collection_plans")

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
        self.results = collection_steps.collect_and_process_symbol(collection_plan)

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
        collection_steps.update_metadata(base_data_path, metadata_updates)

        # Update historical data
        update_result = collection_steps.update_historical_data(
            base_data_path, collected_data
        )

        if update_result["status"] == "success":
            print("📈 Updated historical dataset:")
            print(f"  • Added {update_result['new_records']:,} new records")
            print(f"  • Updated {update_result['symbols_updated']} symbols")
            print(f"  • Total records: {update_result['total_records']:,}")

        # Store collection results for W&B logging
        self.collection_results = update_result
        self.metadata_updates = metadata_updates
        self.collected_data_count = len(collected_data)

        print("✅ Successfully completed batch collection")
        self.next(self.log_artifacts)

    @wandb_log(datasets=True, models=False, others=True, settings=wandb.Settings(project="stock-market-analytics", run_job_type="data-collection"))
    @step
    def log_artifacts(self) -> None:
        """
        Log collection metadata and results to Weights & Biases.

        The wandb_log decorator automatically logs any self attributes as W&B artifacts.
        This includes the collection metadata and summary statistics.
        """
        print("📦 Logging collection metadata to Weights & Biases...")

        try:
            # Store new metadata file:
            base_data_path = Path(os.environ["BASE_DATA_PATH"])
            metadata_file = config.data_collection.metadata_file
            metadata_path = base_data_path / metadata_file

            self.metadata_info = pd.read_csv(metadata_path)

            collection_results = self.collection_results or {}
            self.collection_results = collection_results

            # Collection summary metrics
            self.collection_summary = {
                "total_tickers_processed": len(self.metadata_info['symbol'].unique()),
                "successful_data_collections": self.collected_data_count,
                "metadata_updates_processed": len(self.metadata_updates),
                "new_records_added": self.collection_results.get("new_records", 0),
                "symbols_updated": self.collection_results.get("symbols_updated", 0),
                "total_records_in_dataset": self.collection_results.get("total_records", 0),
                "collection_status": self.collection_results.get("status", "unknown"),
            }

            print("✅ Collection metadata logged successfully")

        except Exception as e:
            print(f"❌ Error logging collection metadata: {e}")
            raise

        self.next(self.end)

    @step
    def end(self) -> None:
        """
        Final step of the batch collection flow.

        This step marks the completion of the entire data collection pipeline.
        All data has been collected, processed, and persisted to the data directory.
        """
        print("✅ Batch Collection Flow completed successfully!")
        print("All stock data has been collected and updated.")


if __name__ == "__main__":
    # Entry point for running the flow directly
    BatchCollectionFlow()

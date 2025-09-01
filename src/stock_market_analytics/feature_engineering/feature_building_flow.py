import os
from pathlib import Path

import pandas as pd
import polars as pl
from hamilton import driver
from metaflow import FlowSpec, step

from stock_market_analytics.feature_engineering import (
    features,
    features_config,
    preprocessing,
)

# Constants
STOCKS_HISTORY_FILE = "stocks_history.parquet"
FEATURES_FILE = "stock_history_features.parquet"


class FeatureBuildingFlow(FlowSpec):
    """
    A Metaflow flow to build features for stock market analytics.
    """

    @step
    def start(self) -> None:
        """
        This is the entry point for the Metaflow pipeline. It validates the
        environment and begins the feature engineering process.
        """
        print("🚀 Starting Feature Engineering Flow...")

        # Validate required environment variables
        if not os.environ.get("BASE_DATA_PATH"):
            raise ValueError("BASE_DATA_PATH environment variable is required")

        print(f"📁 Data directory: {os.environ['BASE_DATA_PATH']}")
        self.next(self.load_inputs)

    @step
    def load_inputs(self) -> None:
        """
        Load input data for feature engineering.
        """
        base_data_path = Path(os.environ["BASE_DATA_PATH"])
        self.data = self._load_stock_data(base_data_path)
        self.next(self.build_features)

    def _load_stock_data(self, base_data_path: Path) -> pl.DataFrame:
        """Load and validate stock data from Parquet file."""
        stocks_history_path = base_data_path / STOCKS_HISTORY_FILE

        if not stocks_history_path.exists():
            raise FileNotFoundError(
                f"Stocks history file not found at {stocks_history_path}. "
                "Stock data must be provided."
            )

        try:
            return pl.read_parquet(stocks_history_path)
        except Exception as e:
            raise ValueError(f"Error loading stocks history file: {str(e)}") from e

    @step
    def build_features(self) -> None:
        """
        Build features from raw stock market data.
        """
        dr = driver.Builder().with_modules(features, preprocessing).build()

        results = dr.execute(
            final_vars=["df_features"],
            inputs={"raw_df": self.data, **features_config},
        )

        past_horizon = features_config.get("past_horizon", 0)

        if past_horizon > 0:
            max_lookback_date = self.data["date"].max() - pd.Timedelta(
                days=past_horizon
            ) #type: ignore
            data = results["df_features"].filter(pl.col("date") >= max_lookback_date)
        else:
            data = results["df_features"]

        self.data = data

        self.next(self.save_features)

    @step
    def save_features(self) -> None:
        """
        Save the engineered features to a parquet file.
        """
        base_data_path = Path(os.environ["BASE_DATA_PATH"])
        features_path = base_data_path / FEATURES_FILE

        self.data.write_parquet(features_path)
        print(f"Features saved to {features_path}")
        self.next(self.end)

    @step
    def end(self) -> None:
        """
        End step: Flow completed.
        """
        print("Feature building flow completed.")


if __name__ == "__main__":
    # Entry point for running the flow directly
    FeatureBuildingFlow()

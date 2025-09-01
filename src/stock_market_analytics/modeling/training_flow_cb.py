import os
from pathlib import Path

import pandas as pd
from hamilton import driver
from metaflow import FlowSpec, step

from stock_market_analytics.modeling import processing_functions

from dataclasses import dataclass
from typing import Iterator, Tuple, List, Optional
from catboost import CatBoostRegressor, Pool


# Constants
FEATURES_FILE = "stock_history_features.parquet"
FEATURES = [
    'dollar_volume',
    'long_kurtosis',
    'short_kurtosis',
    'long_skewness',
    'short_skewness',
    'long_mean',
    'short_mean',
    'mean_diff',
    'long_diff',
    'short_diff',
    'long_short_momentum',
    'pct_from_high_long',
    'pct_from_high_short',
    'year',
    'month',
    'day_of_week',
    'day_of_year'
]

class TrainingFlow(FlowSpec):
    """
    A Metaflow flow to train a CatBoost model for stock market analytics.
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
        self.data = self._load_features(base_data_path)
        self.next(self.prepare_data)

    def _load_features(self, base_data_path: Path) -> pd.DataFrame:
        """Load and validate stock data from Parquet file."""
        features_path = base_data_path / FEATURES_FILE

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found at {features_path}. "
                "Features data must be provided."
            )

        try:
            return pd.read_parquet(features_path)
        except Exception as e:
            raise ValueError(f"Error loading features file: {str(e)}") from e

    @step
    def prepare_data(self) -> None:
        """
        Clean and preprocess the loaded feature data.
        """
        print("🧹 Preparing Feature Data...")
        # We only need to drop null values, since the features are already clean
        df = self.data.dropna()

        dr = driver.Builder().with_modules(processing_functions).build()

        results = dr.execute(
            final_vars=[
                "pools",
                "metadata"
            ],
            inputs={"df": df, "time_span": 180, "features": FEATURES},
        )

        self.pools = results['pools']
        self.metadata = results['metadata']
        self.next(self.end)

    @step
    def end(self) -> None:
        """
        This is the final step of the Metaflow pipeline. It can be used to
        perform any final actions or cleanup.
        """
        print("✅ Feature Engineering Flow completed.")
        print(f"📊 Metadata: {self.metadata.keys()}")
        print(f"🧪 Pools: {self.pools}")

if __name__ == '__main__':
    TrainingFlow()

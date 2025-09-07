import os
from pathlib import Path

from metaflow import FlowSpec, step

from stock_market_analytics.feature_engineering import feature_steps


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
        self.data = feature_steps.load_stock_data(base_data_path)
        self.next(self.build_features)

    @step
    def build_features(self) -> None:
        """
        Build features from raw stock market data.
        """
        base_data_path = Path(os.environ["BASE_DATA_PATH"])

        # Use service function for complete workflow
        result = feature_steps.build_features_from_data(base_data_path)

        print("✅ Feature engineering completed:")
        print(f"  • Processed {result['input_records']:,} input records")
        print(f"  • Generated {result['output_records']:,} feature records")
        print(f"  • Saved to {result['features_file']}")

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

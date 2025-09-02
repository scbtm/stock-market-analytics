import os
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor, Pool
from hamilton import driver
from metaflow import FlowSpec, step

import wandb
from stock_market_analytics.modeling import processing_functions
from stock_market_analytics.modeling.modeling_config import modeling_config
from stock_market_analytics.modeling.modeling_functions import eval_multiquantile
from wandb.integration.metaflow import wandb_log

wandb.login(key=os.environ.get("WANDB_KEY"))

# Constants
FEATURES_FILE = modeling_config["FEATURES_FILE"]
QUANTILES = modeling_config["QUANTILES"]
FEATURES = modeling_config["FEATURES"]
PARAMS = modeling_config["PARAMS"]


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
        self.next(self.training_stage_1)

    def _load_features(self, base_data_path: Path) -> pd.DataFrame:
        """Load and validate stock data from Parquet file."""
        features_path = base_data_path / FEATURES_FILE  # type: ignore

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
    @wandb_log(
        datasets=True,
        models=True,
        others=True,
        settings=wandb.Settings(project="stock-market-analytics"),
    )
    def training_stage_1(self) -> None:
        data = self._prepare_data()

        pools = data["pools"]
        dataset = data["dataset"]

        params = PARAMS.copy()

        early_stopping_rounds = params["num_boost_round"] * 0.08

        # Initialize Catboost regressor with current hyperparameters
        model = CatBoostRegressor(**params)
        
        # Train the regressor
        model.fit(
            pools["train_pool"],
            eval_set=pools["validation_pool"],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
            plot=False,
        )

        final_iterations = model.best_iteration_

        params.update({
            "num_boost_round": final_iterations,
        })

        # Evaluation metrics
        preds = model.predict(pools["validation_pool"])
        ytrue = pools["validation_pool"].get_label()

        print(f"Preds shape: {preds.shape}")
        print(f"Ytrue shape: {ytrue.shape}")

        loss, metrics = eval_multiquantile(
                y_true=ytrue, q_pred=preds, quantiles=QUANTILES, interval=(0.1, 0.9)
            )

        # Log information
        self.loss = loss
        self.metrics = metrics
        self.params = params
        self.model = model
        self.dataset = dataset

        self.next(self.end)

    def _prepare_data(self) -> tuple[dict[str, Pool], dict[str, Any]]:
        """
        Clean and preprocess the loaded feature data.
        """
        print("🧹 Preparing Feature Data...")
        # We only need to drop null values, since the features are already clean
        df = self.data.dropna()

        dr = driver.Builder().with_modules(processing_functions).build()

        data = dr.execute(
            final_vars=["pools", "metadata", "dataset"],
            inputs={"df": df, "time_span": 180, "features": FEATURES},
        )

        return data


    @step
    def end(self) -> None:
        """
        This is the final step of the Metaflow pipeline. It can be used to
        perform any final actions or cleanup.
        """
        print("✅ Feature Engineering Flow completed.")
        print(f"🏆 Best trial: {self.loss}")
        print(f"📊 Evaluation metrics: {self.metrics}")
        print(f"🛠️ Model parameters: {self.params}")
        print(f"🗂️ Model object: {self.model}")


if __name__ == "__main__":
    TrainingFlow()

import os
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from hamilton import driver
from metaflow import FlowSpec, step
from optuna import Trial

import wandb
from stock_market_analytics.modeling import processing_functions
from stock_market_analytics.modeling.modeling_config import modeling_config
from stock_market_analytics.modeling.modeling_functions import eval_multiquantile
from wandb.integration.metaflow import wandb_log

wandb.login(key=os.environ.get("WANDB_KEY"))

# Constants
FEATURES_FILE = modeling_config["FEATURES_FILE"]
QUANTILES = modeling_config["QUANTILES"]
TIMEOUT_MINS = modeling_config["TIMEOUT_MINS"]
N_TRIALS = modeling_config["N_TRIALS"]
STUDY_NAME = modeling_config["STUDY_NAME"]
FEATURES = modeling_config["FEATURES"]


class TuningFlow(FlowSpec):
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
        self.next(self.hyperparameter_tuning)

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
        datasets=False,
        models=False,
        others=True,
        settings=wandb.Settings(project="stock-market-analytics"),
    )
    def hyperparameter_tuning(self) -> None:
        datasets = self._prepare_data()

        pools = datasets["pools"]
        metadata = datasets["metadata"]

        objective_fn = self._get_objective_fn()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=1),
            study_name=STUDY_NAME,
        )
        study.optimize(
            lambda trial: objective_fn(trial, pools),
            n_trials=N_TRIALS,
            timeout=TIMEOUT_MINS * 60,  # seconds
            n_jobs=-1,
        )

        study.set_user_attr("metadata", metadata)

        self.study = study

        self.next(self.end)

    def _prepare_data(self) -> tuple[dict[str, Pool], dict[str, Any]]:
        """
        Clean and preprocess the loaded feature data.
        """
        print("🧹 Preparing Feature Data...")
        # We only need to drop null values, since the features are already clean
        df = self.data.dropna()

        dr = driver.Builder().with_modules(processing_functions).build()

        datasets = dr.execute(
            final_vars=["pools", "metadata"],
            inputs={"df": df, "time_span": 180, "features": FEATURES},
        )

        return datasets

    def _get_objective_fn(self) -> Any:
        """
        Get the objective function for model optimization.
        """

        def objective_fn(trial: Trial, pools: dict[str, Pool]) -> float:
            train_pool = pools["train_pool"]
            val_pool = pools["validation_pool"]

            # Define hyperparameters optimized for stock return prediction with large dataset
            params = {}

            alpha_str = ",".join([str(q) for q in QUANTILES])

            params["loss_function"] = f"MultiQuantile:alpha={alpha_str}"
            params["random_state"] = 1

            # Speed and performance optimizations for large dataset
            # Higher learning rates for faster convergence (stock data has weak signals, avoid too slow learning)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.05, 0.25)

            # Shallower trees to prevent overfitting on noisy financial data and speed up training
            params["depth"] = trial.suggest_int("depth", 4, 7)

            # Stronger L2 regularization for noisy financial data
            params["l2_leaf_reg"] = trial.suggest_int("l2_leaf_reg", 1, 15)

            # Remove slower grow policies, keep faster ones
            params["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise"]
            )

            # Keep num_boost_round fixed for early stopping
            params["num_boost_round"] = 1_000

            # Optimize bootstrap types for large datasets (removed MVS which can be slower)
            params["bootstrap_type"] = trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli"]
            )

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0.5, 8
                )

            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.6, 0.95)

            # Add column sampling for better generalization and speed
            params["colsample_bylevel"] = trial.suggest_float(
                "colsample_bylevel", 0.6, 1.0
            )

            # Optimize for large datasets
            params["border_count"] = trial.suggest_int(
                "border_count", 128, 254
            )  # Higher for large datasets

            # Add min_data_in_leaf for regularization on large datasets
            params["min_data_in_leaf"] = 100

            params["verbose"] = False

            # More aggressive early stopping for faster tuning
            early_stopping_rounds = int(
                params["num_boost_round"] * 0.08
            )  # Reduced from 0.1 to 0.08

            # Initialize Catboost regressor with current hyperparameters
            model = CatBoostRegressor(**params)
            # Train the regressor
            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
                plot=False,
            )

            # Evaluation metrics
            preds = model.predict(val_pool)
            ytrue = val_pool.get_label()

            loss, metrics = eval_multiquantile(
                y_true=ytrue, q_pred=preds, quantiles=QUANTILES, interval=(0.1, 0.9)
            )

            trial.set_user_attr("metrics", metrics)

            return loss

        return objective_fn

    @step
    def end(self) -> None:
        """
        This is the final step of the Metaflow pipeline. It can be used to
        perform any final actions or cleanup.
        """
        print("✅ Feature Engineering Flow completed.")
        print(f"🏆 Best trial: {self.study.best_trial.value}")


if __name__ == "__main__":
    TuningFlow()

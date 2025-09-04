import os
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from metaflow import FlowSpec, step
from optuna import Trial

import wandb
from stock_market_analytics.modeling import processing_functions
from stock_market_analytics.modeling.pipeline_components.configs import modeling_config
from stock_market_analytics.modeling.pipeline_components.evaluators import (
    ModelEvaluator,
)
from stock_market_analytics.modeling.pipeline_components.pipeline_factory import (
    get_pipeline,
)
from wandb.integration.metaflow import wandb_log

wandb.login(key=os.environ.get("WANDB_KEY"))

# Constants
FEATURES_FILE = modeling_config["FEATURES_FILE"]
QUANTILES = modeling_config["QUANTILES"]
TIMEOUT_MINS = modeling_config["TIMEOUT_MINS"]
N_TRIALS = modeling_config["N_TRIALS"]
STUDY_NAME = modeling_config["STUDY_NAME"]
FEATURES = modeling_config["FEATURES"]
TARGET = modeling_config["TARGET"]
TIME_SPAN = modeling_config["TIME_SPAN"]


class TuningFlow(FlowSpec):
    """
    A Metaflow flow to tune a CatBoost model hyperparameters for stock market analytics.
    """

    @step
    def start(self) -> None:
        """
        This is the entry point for the Metaflow pipeline. It validates the
        environment and begins the tuning process.
        """
        print("🚀 Starting Tuning Flow...")

        # Validate required environment variables
        if not os.environ.get("BASE_DATA_PATH"):
            raise ValueError("BASE_DATA_PATH environment variable is required")

        print(f"📁 Data directory: {os.environ['BASE_DATA_PATH']}")
        self.next(self.load_inputs)

    @step
    def load_inputs(self) -> None:
        """
        Load input data for data processing.
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
        modeling_datasets = self._prepare_data()

        objective_fn = self._get_objective_fn()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=1),
            study_name=STUDY_NAME,
        )
        study.optimize(
            lambda trial: objective_fn(trial, modeling_datasets),
            n_trials=N_TRIALS,
            timeout=TIMEOUT_MINS * 60,  # seconds
            n_jobs=-1,
        )

        metadata = processing_functions.metadata(split_data=self.data)
        study.set_user_attr("metadata", metadata)

        self.study = study

        self.next(self.end)

    def _prepare_data(self) -> dict[str, Any]:
        """
        Clean and preprocess the loaded feature data.
        """
        print("🧹 Preparing Feature Data...")
        # We only need to drop null values, since the features are already clean
        df = self.data.dropna()

        # Split data chronologically
        data = processing_functions.split_data(df=df, time_span=TIME_SPAN)

        # Prepare modeling datasets
        modeling_datasets = processing_functions.modeling_datasets(
            split_data=data,
            features=FEATURES,
            target=TARGET,
        )

        return modeling_datasets

    def _get_objective_fn(self) -> Any:
        """
        Get the objective function for model optimization.
        """

        def objective_fn(trial: Trial, modeling_datasets: dict[str, Any]) -> float:
            xtrain, ytrain = modeling_datasets["xtrain"], modeling_datasets["ytrain"]
            xval, yval = modeling_datasets["xval"], modeling_datasets["yval"]

            # Get base pipeline
            pipeline = get_pipeline()

            # Define hyperparameters optimized for stock return prediction with large dataset
            catboost_params = {}

            # Speed and performance optimizations for large dataset
            # Higher learning rates for faster convergence (stock data has weak signals, avoid too slow learning)
            catboost_params["learning_rate"] = trial.suggest_float("learning_rate", 0.05, 0.25)

            # Shallower trees to prevent overfitting on noisy financial data and speed up training
            catboost_params["depth"] = trial.suggest_int("depth", 4, 7)

            # Stronger L2 regularization for noisy financial data
            catboost_params["l2_leaf_reg"] = trial.suggest_int("l2_leaf_reg", 1, 15)

            # Remove slower grow policies, keep faster ones
            catboost_params["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise"]
            )

            # Keep num_boost_round fixed for early stopping
            catboost_params["num_boost_round"] = 1_000

            # Optimize bootstrap types for large datasets (removed MVS which can be slower)
            catboost_params["bootstrap_type"] = trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli"]
            )

            if catboost_params["bootstrap_type"] == "Bayesian":
                catboost_params["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0.5, 8
                )

            elif catboost_params["bootstrap_type"] == "Bernoulli":
                catboost_params["subsample"] = trial.suggest_float("subsample", 0.6, 0.95)

            # Add column sampling for better generalization and speed
            catboost_params["colsample_bylevel"] = trial.suggest_float(
                "colsample_bylevel", 0.6, 1.0
            )

            # Optimize for large datasets
            catboost_params["border_count"] = trial.suggest_int(
                "border_count", 128, 254
            )  # Higher for large datasets

            # Add min_data_in_leaf for regularization on large datasets
            catboost_params["min_data_in_leaf"] = 100

            catboost_params["verbose"] = False

            # Update the pipeline's CatBoost parameters
            pipeline.named_steps["quantile_regressor"].set_params(**catboost_params)

            # Fit PCA on training data first
            pca = pipeline.named_steps["pca"]
            pca.fit(xtrain)
            _xval = pca.transform(xval)

            # More aggressive early stopping for faster tuning
            early_stopping_rounds = int(catboost_params["num_boost_round"] * 0.08)

            # Prepare fit parameters for early stopping
            fit_params = {
                "early_stopping_rounds": early_stopping_rounds,
                "verbose": False,
                "plot": False,
            }
            fit_params["eval_set"] = (_xval, yval)
            fit_params = {f"quantile_regressor__{k}": v for k, v in fit_params.items()}

            # Fit the pipeline
            pipeline.fit(xtrain, ytrain, **fit_params)

            # Evaluation metrics using the new evaluator
            evaluator = ModelEvaluator()
            loss, metrics = evaluator.evaluate_training(pipeline, xval, yval)

            trial.set_user_attr("metrics", metrics)

            return loss

        return objective_fn

    @step
    def end(self) -> None:
        """
        This is the final step of the Metaflow pipeline. It can be used to
        perform any final actions or cleanup.
        """
        print("✅ Tuning Flow completed.")
        print(f"🏆 Best trial: {self.study.best_trial.value}")


if __name__ == "__main__":
    TuningFlow()

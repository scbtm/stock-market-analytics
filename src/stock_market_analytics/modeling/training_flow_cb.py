import os
from pathlib import Path
from stock_market_analytics.modeling.pipeline_components.parameters import cb_fit_params
from stock_market_analytics.modeling.pipeline_factory import get_pipeline
import pandas as pd
from metaflow import FlowSpec, step

import wandb
from stock_market_analytics.modeling import processing_functions
from stock_market_analytics.modeling.pipeline_components.configs import modeling_config
from stock_market_analytics.modeling.pipeline_components.parameters import cb_model_params, cb_fit_params
from stock_market_analytics.modeling.pipeline_components.evaluators import ModelEvaluator, EvaluationReport
from stock_market_analytics.modeling.pipeline_components.predictors import CatBoostMultiQuantileModel
from wandb.integration.metaflow import wandb_log

wandb.login(key=os.environ.get("WANDB_KEY"))

# Constants
FEATURES_FILE = modeling_config["FEATURES_FILE"]
QUANTILES = modeling_config["QUANTILES"]
FEATURES = modeling_config["FEATURES"]
TARGET_COVERAGE = modeling_config["TARGET_COVERAGE"]
TARGET = modeling_config["TARGET"]
TIME_SPAN = modeling_config["TIME_SPAN"]  # days

class TrainingFlow(FlowSpec):
    """
    A Metaflow flow to train a CatBoost model for stock market analytics.
    """

    @step
    def start(self) -> None:
        """
        This is the entry point for the Metaflow pipeline. It validates the
        environment and begins the model training process.
        """
        print("🚀 Starting model training Flow...")

        # Validate required environment variables
        if not os.environ.get("BASE_DATA_PATH"):
            raise ValueError("BASE_DATA_PATH environment variable is required")

        print(f"📁 Data directory: {os.environ['BASE_DATA_PATH']}")
        self.next(self.load_inputs)

    @step
    def load_inputs(self) -> None:
        """
        Load input data for model training.
        """
        base_data_path = Path(os.environ["BASE_DATA_PATH"])
        data = self._load_features(base_data_path)
        #only for training we remove the nulls
        self.data = data.dropna()
        self.next(self.model_training)

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
    def model_training(self) -> None:
        print("🧹 Preparing Feature Data...")

        # Prepare data
        data = self.data

        data = processing_functions.split_data(df=data, time_span=TIME_SPAN)

        modeling_datasets = processing_functions.modeling_datasets(
            split_data=data,
            features=FEATURES,
            target=TARGET,
        )

        xtrain, ytrain = modeling_datasets["xtrain"], modeling_datasets["ytrain"]
        xval, yval = modeling_datasets["xval"], modeling_datasets["yval"]

        print("🤖 Training CatBoost Quantile Regressor...")

        pipeline = get_pipeline()
        pca = pipeline.named_steps["pca"]  # type: ignore
        pca.fit(xtrain)  # Fit PCA on training data only. 
        # This is needed as there is no gracefull way to handle this in the pipeline if we want to use early stopping.
        _xval = pca.transform(xval)
        # Fit the pipeline with early stopping parameters. This need to be passed to the pipeline as follows:
        fit_params = cb_fit_params.copy()
        fit_params["eval_set"] = (_xval, yval)
        fit_params = {f"quantile_regressor__{k}": v for k, v in fit_params.items()}
        pipeline.fit(xtrain, ytrain, **fit_params)

        quantile_regressor: CatBoostMultiQuantileModel = pipeline.named_steps["quantile_regressor"]  # type: ignore
        final_iterations = quantile_regressor.best_iteration_
        print(f"🏁 Training completed in {final_iterations} iterations.")

        # Evaluation metrics. The pipeline is now fitted with the best model and ready for inference. (although calibration is pending)
        evaluator = ModelEvaluator()
        loss, metrics = evaluator.evaluate_training(pipeline, xval, yval)

        # Pass information to next step
        calibration_info = {
            "loss": loss,
            "metrics": metrics,
            "params": cb_model_params,
            "model": quantile_regressor,
            "modeling_datasets": modeling_datasets,
            "pipeline": pipeline
        }

        self.calibration_info = calibration_info
        self.next(self.calibrate_model)

    @step
    @wandb_log(
        datasets=True,
        models=True,
        others=True,
        settings=wandb.Settings(project="stock-market-analytics"),
        )
    def calibrate_model(self) -> None:
        """
        Perform conformal adjustment on the trained model.
        """

        calibration_info = self.calibration_info

        # model = calibration_info["model"]
        model = calibration_info["pipeline"]

        params = calibration_info["params"]

        modeling_datasets = calibration_info["modeling_datasets"]

        xcal, ycal = modeling_datasets["xval"], modeling_datasets["yval"]
        xtest, ytest = modeling_datasets["xtest"], modeling_datasets["ytest"]

        # Perform conformal evaluation using the evaluator
        evaluator = ModelEvaluator()
        conformal_results = evaluator.evaluate_conformal(model, xcal, ycal, xtest, ytest)

        final_metrics = {
            "coverage": [conformal_results["coverage"]],
            "mean_width": [conformal_results["mean_width"]],
            "pinball_loss": [conformal_results["pinball_loss"]],
        }

        training_metrics = calibration_info["metrics"]

        # Log results to wandb
        self.params = params
        self.pipeline = model
        self.training_metrics = training_metrics
        self.development_data = processing_functions.metadata(split_data=self.data)
        self.final_metrics = final_metrics

        self.next(self.end)


    @step
    def end(self) -> None:
        """
        This is the final step of the Metaflow pipeline. It can be used to
        perform any final actions or cleanup.
        """
        print("✅ Training Flow completed.")
        
        # Display formatted evaluation results
        evaluation_results = {
            "training": {"metrics": self.training_metrics},
            "conformal": self.final_metrics
        }
        EvaluationReport.print_summary(evaluation_results)

if __name__ == "__main__":
    TrainingFlow()

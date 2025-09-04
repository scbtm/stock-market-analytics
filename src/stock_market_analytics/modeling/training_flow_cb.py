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
from stock_market_analytics.modeling.modeling_functions import (
    eval_multiquantile,
    conformal_adjustment,
    apply_conformal,
    mean_width,
    pinball_loss,
    coverage
)
from stock_market_analytics.modeling.pipeline_components.predictors import CatBoostMultiQuantileModel
from wandb.integration.metaflow import wandb_log

wandb.login(key=os.environ.get("WANDB_KEY"))

# Constants
FEATURES_FILE = modeling_config["FEATURES_FILE"]
QUANTILES = modeling_config["QUANTILES"]
FEATURES = modeling_config["FEATURES"]
TARGET_COVERAGE = modeling_config["TARGET_COVERAGE"]
LOW, MID, HIGH = modeling_config["LOW"], modeling_config["MID"], modeling_config["HIGH"]
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
        pca.fit(xtrain)  # Fit PCA on training data only
        _xval = pca.transform(xval)
        # Fit the pipeline with early stopping parameters. This need to be passed to the pipeline as follows:
        fit_params = cb_fit_params.copy()
        fit_params["eval_set"] = (_xval, yval)
        fit_params = {f"quantile_regressor__{k}": v for k, v in fit_params.items()}
        pipeline.fit(xtrain, ytrain, **fit_params)

        quantile_regressor: CatBoostMultiQuantileModel = pipeline.named_steps["quantile_regressor"]  # type: ignore
        final_iterations = quantile_regressor.best_iteration_
        print(f"🏁 Training completed in {final_iterations} iterations.")

        # Evaluation metrics using new wrapper
        preds = pipeline.predict(xval)
        ytrue = yval.values

        loss, metrics = eval_multiquantile(
                y_true=ytrue, q_pred=preds, quantiles=QUANTILES, interval=(0.1, 0.9)
            )

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

        xcal, ycal = modeling_datasets["xval"], modeling_datasets["yval"].values
        xtest, ytest = modeling_datasets["xtest"], modeling_datasets["ytest"].values

        # Predict quantiles using sklearn wrapper
        q_cal = model.predict(xcal)
        q_tst = model.predict(xtest)

        # Pull specific lower/upper for CQR (here 10% / 90%)
        qlo_cal, qhi_cal = q_cal[:, LOW], q_cal[:, HIGH]
        qlo_tst, qhi_tst = q_tst[:, LOW], q_tst[:, HIGH]

        # Conformal adjustment on calibration slice
        qconf = conformal_adjustment(qlo_cal, qhi_cal, ycal, alpha=1 - TARGET_COVERAGE)

        # Apply to test
        lo_cqr, hi_cqr = apply_conformal(qlo_tst, qhi_tst, qconf)
        med_pred = q_tst[:, MID]

        # Metrics
        cov   = coverage(ytest, lo_cqr, hi_cqr)
        width = mean_width(lo_cqr, hi_cqr)
        pin50 = pinball_loss(ytest, med_pred, alpha=0.5)

        final_metrics = {
            "coverage": [cov],
            "mean_width": [width],
            "pinball_loss": [pin50],
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
        print(f"🛠️ Training Metrics: {self.training_metrics}")
        print(f"📊 Evaluation Metrics: {self.final_metrics}")

if __name__ == "__main__":
    TrainingFlow()

import os
from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, step

import wandb
from stock_market_analytics.modeling import processing_functions
from stock_market_analytics.modeling.pipeline_components.calibrators import (
    PipelineWithCalibrator,
)
from stock_market_analytics.modeling.pipeline_components.configs import modeling_config
from stock_market_analytics.modeling.pipeline_components.evaluators import (
    EvaluationReport,
    ModelEvaluator,
)
from stock_market_analytics.modeling.pipeline_components.parameters import (
    cb_fit_params,
)
from stock_market_analytics.modeling.pipeline_components.pipeline_factory import (
    get_pipeline,
    get_baseline_pipeline,
)
from stock_market_analytics.modeling.pipeline_components.predictors import (
    CatBoostMultiQuantileModel,
)
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
        transformations = pipeline[0]  # type: ignore
        transformations.fit(xtrain)  # Fit PCA on training data only.
        # This is needed as there is no gracefull way to handle this in the pipeline if we want to use early stopping.
        _xval = transformations.transform(xval)
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
            "modeling_datasets": modeling_datasets,
            "pipeline": pipeline
        }

        self.data = data  # Pass the original data with fold column for logging

        self.calibration_info = calibration_info
        self.next(self.baseline_training, self.calibrate_model)

    @step
    def baseline_training(self) -> None:
        """
        Train baseline quantile regressors for comparison.
        """
        print("📊 Training baseline quantile regressors...")
        
        calibration_info = self.calibration_info
        modeling_datasets = calibration_info["modeling_datasets"]
        
        xtrain, ytrain = modeling_datasets["xtrain"], modeling_datasets["ytrain"]
        xval, yval = modeling_datasets["xval"], modeling_datasets["yval"]
        
        evaluator = ModelEvaluator()
        
        # Train and evaluate different baselines
        baselines = ["historical"]
        baseline_results = {}
        
        for baseline_name in baselines:
            print(f"🔄 Training {baseline_name} baseline...")
            
            # Get baseline pipeline
            baseline_pipeline = get_baseline_pipeline(baseline_name)
            
            # Train the baseline (no early stopping needed for simple baselines)
            baseline_pipeline.fit(xtrain, ytrain)
            
            # Evaluate on validation set
            loss, metrics = evaluator.evaluate_training(baseline_pipeline, xval, yval)
            
            baseline_results[baseline_name] = {
                "pipeline": baseline_pipeline,
                "loss": loss,
                "metrics": metrics
            }
            
            print(f"✅ {baseline_name} baseline - Loss: {loss:.4f}")
        
        # Store baseline results
        self.baseline_results = baseline_results
        
        # Continue to join step
        self.next(self.join_results)

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
        data = self.data

        # model = calibration_info["model"]
        pipeline = calibration_info["pipeline"]

        modeling_datasets = calibration_info["modeling_datasets"]

        xcal, ycal = modeling_datasets["xval"], modeling_datasets["yval"]
        xtest, ytest = modeling_datasets["xtest"], modeling_datasets["ytest"]

        # Create calibrated pipeline for production use
        print("🔧 Creating calibrated pipeline...")
        calibrated_pipeline, calibrator = PipelineWithCalibrator.create_calibrated_pipeline(
            base_pipeline=pipeline,
            X_cal=xcal,
            y_cal=ycal
        )

        # Evaluate calibrated predictions (independent of calibrator)
        evaluator = ModelEvaluator()

        # Get calibrated bounds and evaluate them
        calibrated_bounds = calibrated_pipeline.predict(xtest)

        # Get median predictions for pinball loss
        raw_predictions = pipeline.predict(xtest)
        mid_idx = modeling_config["MID"]
        median_predictions = raw_predictions[:, mid_idx]

        conformal_results = evaluator.evaluate_calibrated_predictions(
            calibrated_bounds, ytest, median_predictions
        )

        final_metrics = {
            "coverage": [conformal_results["coverage"]],
            "mean_width": [conformal_results["mean_width"]],
            "pinball_loss": [conformal_results["pinball_loss"]],
        }

        print(f"📏 Conformal quantile: {calibrator.conformal_quantile_:.4f}")
        print(f"🎯 Target coverage: {calibrator.target_coverage:.1%}")

        training_metrics = calibration_info["metrics"]

        # Log results to wandb
        self.calibrated_pipeline = calibrated_pipeline  # Production-ready pipeline with conformal calibration
        self.training_metrics = training_metrics
        self.data = data
        self.final_metrics = final_metrics

        self.next(self.join_results)

    @step 
    def join_results(self, inputs: list) -> None:
        """
        Join results from CatBoost training and baseline training branches.
        """
        print("🔀 Joining training results...")
        
        # Merge artifacts from both branches
        # Find the calibrate_model input (has calibrated_pipeline)
        # Find the baseline_training input (has baseline_results)
        
        catboost_input = None
        baseline_input = None
        
        for inp in inputs:
            if hasattr(inp, 'calibrated_pipeline'):
                catboost_input = inp
            if hasattr(inp, 'baseline_results'):
                baseline_input = inp
        
        if catboost_input is None or baseline_input is None:
            raise ValueError("Could not find both CatBoost and baseline results")
        
        # Copy all CatBoost results  
        self.calibrated_pipeline = catboost_input.calibrated_pipeline
        self.training_metrics = catboost_input.training_metrics
        self.data = catboost_input.data
        self.final_metrics = catboost_input.final_metrics
        
        # Add baseline results
        self.baseline_results = baseline_input.baseline_results
        
        print("✅ Results joined successfully")
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

        epsilon = 1e-8  # Small value to prevent division by zero

        # Display calibrator information
        print("\n🔧 Calibrated Pipeline Summary:")
        print(f"Pipeline steps: {[step[0] for step in self.calibrated_pipeline.steps]}")
        calibrator_info = self.calibrated_pipeline[-1].get_conformal_info()
        print(f"Conformal quantile: {calibrator_info['conformal_quantile']:.4f}")
        print(f"Target coverage: {calibrator_info['target_coverage']:.1%}")

        # Display baseline comparison
        if hasattr(self, 'baseline_results'):
            print(f"\n📊 Baseline Model Comparison:")
            print(f"{'Model':<15} {'Validation Loss':<15} {'Coverage':<12} {'Width':<12}")
            print(f"{'-'*15} {'-'*15} {'-'*12} {'-'*12}")
            
            # CatBoost results
            catboost_loss = self.training_metrics.get('pinball_mean', 0.0)
            catboost_coverage = self.training_metrics.get('coverage_10_90', 0.0) 
            catboost_width = self.training_metrics.get('mean_width', 0.0)
            print(f"{'CatBoost':<15} {catboost_loss:<15.4f} {catboost_coverage:<12.3f} {catboost_width:<12.3f}")
            
            # Baseline results
            for baseline_name, baseline_data in self.baseline_results.items():
                loss = baseline_data['loss']
                metrics = baseline_data['metrics']
                coverage = metrics.get('coverage_10_90', 0.0)
                width = metrics.get('mean_width', 0.0)
                print(f"{baseline_name.capitalize():<15} {loss:<15.4f} {coverage:<12.3f} {width:<12.3f}")
            
            # Performance comparison
            best_baseline_loss = min([data['loss'] for data in self.baseline_results.values()])
            catboost_loss_val = self.training_metrics.get('pinball_mean', float('inf'))
            
            if catboost_loss_val < best_baseline_loss:
                improvement = ((best_baseline_loss - catboost_loss_val) / best_baseline_loss + epsilon) * 100
                print(f"\n🎯 CatBoost improvement over best baseline: {improvement:.1f}%")
            else:
                degradation = ((catboost_loss_val - best_baseline_loss) / best_baseline_loss + epsilon) * 100
                print(f"\n⚠️  CatBoost performance vs best baseline: -{degradation:.1f}%")

        # Usage example
        print(f"\n💡 Usage:")
        print(f"# For raw quantile predictions: Use main pipeline")
        print(f"# For conformal bounds: self.calibrated_pipeline.predict(X)")
        print(f"# For baseline comparisons: self.baseline_results[<baseline_name>]['pipeline'].predict(X)")
        print(f"# Returns: array of shape (n_samples, 2) for conformal, (n_samples, n_quantiles) for others")

if __name__ == "__main__":
    TrainingFlow()

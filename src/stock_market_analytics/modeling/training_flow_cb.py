import os
from pathlib import Path

from metaflow import FlowSpec, step

import wandb
from stock_market_analytics.modeling import modeling_steps
from stock_market_analytics.modeling.model_factory.evaluation.evaluators import (
    EvaluationReport,
)
from wandb.integration.metaflow import wandb_log

wandb.login(key=os.environ.get("WANDB_KEY"))


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
        data = modeling_steps.load_features(base_data_path)
        # only for training we remove the nulls
        self.data = data.dropna()
        self.next(self.model_training)


    @step
    def model_training(self) -> None:
        """Main training step that orchestrates the model training process."""
        print("🧹 Preparing Feature Data...")

        # Prepare data for training
        data_prep_result = modeling_steps.prepare_training_data(self.data)
        modeling_datasets = data_prep_result["modeling_datasets"]
        self.data = data_prep_result["split_data"]

        print("🤖 Training CatBoost Quantile Regressor...")

        # Train the model
        pipeline, final_iterations = modeling_steps.train_catboost_model(modeling_datasets)

        # Analyze and display feature importance
        feature_importance_df = modeling_steps.analyze_feature_importance(pipeline, modeling_datasets["xtrain"])
        print("📊 Feature Importances:")
        print(feature_importance_df)

        print(f"🏁 Training completed in {final_iterations + 1} iterations.")

        # Evaluate the trained model
        loss, metrics = modeling_steps.evaluate_model(pipeline, modeling_datasets)

        # Prepare data for next steps
        calibration_info = {
            "loss": loss,
            "metrics": metrics,
            "modeling_datasets": modeling_datasets,
            "pipeline": pipeline,
        }

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

        # Train baseline models using modeling steps
        baseline_results = modeling_steps.train_baseline_models(modeling_datasets)

        # Display results
        for baseline_name, baseline_data in baseline_results.items():
            print(f"✅ {baseline_name} baseline - Loss: {baseline_data['loss']:.4f}")

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
        pipeline = calibration_info["pipeline"]
        modeling_datasets = calibration_info["modeling_datasets"]

        # Create calibrated pipeline for production use
        print("🔧 Creating calibrated pipeline...")
        calibrated_pipeline, calibrator = modeling_steps.create_calibrated_pipeline(
            pipeline, modeling_datasets
        )

        # Evaluate calibrated predictions
        final_metrics = modeling_steps.evaluate_calibrated_predictions(
            calibrated_pipeline, pipeline, modeling_datasets
        )

        print(f"📏 Conformal quantile: {calibrator.conformal_quantile_:.4f}")
        print(f"📊 Target coverage: {calibrator.target_coverage:.1%}")

        training_metrics = calibration_info["metrics"]

        # Log results to wandb
        self.calibrated_pipeline = (
            calibrated_pipeline  # Production-ready pipeline with conformal calibration
        )
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
            if hasattr(inp, "calibrated_pipeline"):
                catboost_input = inp
            if hasattr(inp, "baseline_results"):
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
            "conformal": self.final_metrics,
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
        if hasattr(self, "baseline_results"):
            print("\n📊 Baseline Model Comparison:")
            print(
                f"{'Model':<15} {'Validation Loss':<15} {'Coverage':<12} {'Width':<12}"
            )
            print(f"{'-' * 15} {'-' * 15} {'-' * 12} {'-' * 12}")

            # CatBoost results
            catboost_loss = self.training_metrics.get("pinball_mean", 0.0)
            catboost_coverage = self.training_metrics.get("coverage_10_90", 0.0)
            catboost_width = self.training_metrics.get("mean_width", 0.0)
            print(
                f"{'CatBoost':<15} {catboost_loss:<15.4f} {catboost_coverage:<12.3f} {catboost_width:<12.3f}"
            )

            # Baseline results
            for baseline_name, baseline_data in self.baseline_results.items():
                loss = baseline_data["loss"]
                metrics = baseline_data["metrics"]
                coverage = metrics.get("coverage_10_90", 0.0)
                width = metrics.get("mean_width", 0.0)
                print(
                    f"{baseline_name.capitalize():<15} {loss:<15.4f} {coverage:<12.3f} {width:<12.3f}"
                )

            # Performance comparison
            best_baseline_loss = min(
                [data["loss"] for data in self.baseline_results.values()]
            )
            catboost_loss_val = self.training_metrics.get("pinball_mean", float("inf"))

            if catboost_loss_val < best_baseline_loss:
                improvement = (
                    (best_baseline_loss - catboost_loss_val) / best_baseline_loss
                    + epsilon
                ) * 100
                print(
                    f"\n🏆CatBoost improvement over best baseline: {improvement:.1f}%"
                )
            else:
                degradation = (
                    (catboost_loss_val - best_baseline_loss) / best_baseline_loss
                    + epsilon
                ) * 100
                print(
                    f"\n⚠️  CatBoost performance vs best baseline: -{degradation:.1f}%"
                )

        # Usage example
        print("\n💡 Usage:")
        print("# For raw quantile predictions: Use main pipeline")
        print("# For conformal bounds: self.calibrated_pipeline.predict(X)")
        print(
            "# For baseline comparisons: self.baseline_results[<baseline_name>]['pipeline'].predict(X)"
        )
        print(
            "# Returns: array of shape (n_samples, 2) for conformal, (n_samples, n_quantiles) for others"
        )


if __name__ == "__main__":
    TrainingFlow()

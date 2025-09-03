"""
Refactored training flow using clean components.

This module provides a clean, maintainable training flow that delegates
business logic to reusable components while focusing on orchestration.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict

from metaflow import FlowSpec, step
import wandb
from wandb.integration.metaflow import wandb_log

from ..components import CatBoostMultiQuantileModel, DataProcessor, MultiQuantileEvaluator
from ..config import ConfigManager
from ..inference import ProductionPredictor, ModelLoader

logger = logging.getLogger(__name__)

# Initialize wandb
wandb.login(key=os.environ.get("WANDB_KEY"))


class TrainingFlow(FlowSpec):
    """
    Clean training flow for CatBoost multi-quantile models.
    
    This flow orchestrates the training process using reusable components,
    keeping the flow logic focused on coordination rather than implementation details.
    """

    @step
    def start(self) -> None:
        """
        Initialize the training flow and validate environment.
        """
        print("🚀 Starting clean training flow...")

        # Validate required environment variables
        if not os.environ.get("BASE_DATA_PATH"):
            raise ValueError("BASE_DATA_PATH environment variable is required")

        self.base_data_path = Path(os.environ["BASE_DATA_PATH"])
        print(f"📁 Data directory: {self.base_data_path}")

        # Initialize configuration (can be extended to load from file)
        self.config = ConfigManager()
        print(f"⚙️  Configuration: {self.config}")

        self.next(self.load_and_prepare_data)

    @step
    def load_and_prepare_data(self) -> None:
        """
        Load and prepare data using the DataProcessor component.
        """
        print("📊 Loading and preparing data...")

        # Initialize data processor
        data_processor = DataProcessor(
            features=self.config.data.features,
            target=self.config.data.target
        )

        # Load features
        raw_data = data_processor.load_features(
            data_path=self.base_data_path,
            features_file=self.config.data.features_file
        )

        # Clean data
        clean_data = data_processor.clean_data(raw_data)

        # Prepare data splits
        data_splits = data_processor.prepare_data_splits(
            df=clean_data,
            time_span=self.config.data.time_span
        )

        # Store results
        self.pools = data_splits["pools"]
        self.metadata = data_splits["metadata"]
        self.dataset = data_splits["dataset"]

        print(f"✅ Data prepared: {self.metadata['training_n_rows']} train, "
              f"{self.metadata['validation_n_rows']} validation, "
              f"{self.metadata['test_n_rows']} test samples")

        self.next(self.train_model)

    @step
    def train_model(self) -> None:
        """
        Train the CatBoost model using the clean model wrapper.
        """
        print("🧠 Training CatBoost model...")

        # Initialize model
        model = CatBoostMultiQuantileModel(
            quantiles=self.config.model.catboost.quantiles,
            random_state=self.config.model.catboost.random_state,
            **self.config.model.catboost.to_catboost_params()
        )

        # Calculate early stopping rounds
        early_stopping_rounds = int(
            self.config.model.catboost.num_boost_round * 0.08
        )

        # Train model
        model.fit(
            train_pool=self.pools["train_pool"],
            eval_pool=self.pools["validation_pool"],
            early_stopping_rounds=early_stopping_rounds,
            verbose=int(early_stopping_rounds / 2)
        )

        self.trained_model = model

        # Log training completion
        print(f"✅ Model training completed. Best iteration: {model.best_iteration}")

        self.next(self.evaluate_baselines)

    @step
    def evaluate_baselines(self) -> None:
        """
        Evaluate baseline predictors for comparison.
        """
        print("📊 Evaluating baseline predictors...")

        # Initialize evaluator
        evaluator = MultiQuantileEvaluator(
            quantiles=self.config.model.catboost.quantiles,
            target_coverage=self.config.evaluation.target_coverage,
            coverage_interval=self.config.evaluation.coverage_interval
        )

        # Evaluate trained model on validation set
        trained_metrics = evaluator.evaluate_model(
            model=self.trained_model,
            validation_pool=self.pools["validation_pool"],
            return_predictions=False
        )

        # Evaluate baseline predictors
        baseline_results = evaluator.evaluate_baselines(
            train_pool=self.pools["train_pool"],
            validation_pool=self.pools["validation_pool"],
            baseline_strategies=self.config.evaluation.baseline_strategies
        )

        self.trained_metrics = trained_metrics
        self.baseline_results = baseline_results

        print(f"✅ Baseline evaluation completed for {len(baseline_results)} strategies")

        self.next(self.calibrate_conformal)

    @step
    @wandb_log(
        datasets=True,
        models=True,
        others=True,
        settings=wandb.Settings(project="stock-market-analytics"),
    )
    def calibrate_conformal(self) -> None:
        """
        Perform conformal prediction calibration and final evaluation.
        """
        print("🎯 Performing conformal prediction calibration...")

        # Initialize evaluator
        evaluator = MultiQuantileEvaluator(
            quantiles=self.config.model.catboost.quantiles,
            target_coverage=self.config.evaluation.target_coverage,
            coverage_interval=self.config.evaluation.coverage_interval
        )

        # Prepare calibration and test data
        calibration_data = self.dataset[self.dataset['fold'] == 'validation'].copy()
        test_data = self.dataset[self.dataset['fold'] == 'test'].copy()

        # Compute conformal adjustment
        conformal_adjustment = evaluator.calibrate_conformal(
            model=self.trained_model,
            calibration_data=calibration_data,
            features=self.config.data.features,
            target=self.config.data.target
        )

        # Evaluate on test set with conformal prediction
        conformal_results = evaluator.evaluate_conformal(
            model=self.trained_model,
            test_data=test_data,
            features=self.config.data.features,
            target=self.config.data.target,
            q_conformal=conformal_adjustment
        )

        # Compare with baselines
        comparison_results = evaluator.compare_with_baselines(
            trained_metrics=conformal_results["metrics"],
            baseline_results=self.baseline_results,
            test_data=test_data,
            features=self.config.data.features,
            q_conformal=conformal_adjustment
        )

        # Store results for logging
        self.conformal_adjustment = conformal_adjustment
        self.conformal_results = conformal_results
        self.comparison_results = comparison_results

        # Create production predictor
        self.production_predictor = ModelLoader.create_predictor_from_training(
            trained_model=self.trained_model,
            config=self.config,
            conformal_adjustment=conformal_adjustment,
            training_metrics=self.trained_metrics,
            training_params=self.config.model.catboost.to_catboost_params()
        )

        print(f"✅ Conformal calibration completed. Adjustment: {conformal_adjustment:.4f}")
        print(f"📊 Final coverage: {conformal_results['metrics']['coverage']:.3f} "
              f"(target: {self.config.evaluation.target_coverage:.3f})")

        self.next(self.end)

    @step
    def end(self) -> None:
        """
        Finalize the training flow and display results.
        """
        print("🎉 Training flow completed successfully!")

        # Display final metrics
        metrics = self.conformal_results["metrics"]
        print("\n📈 Final Model Performance:")
        print("-" * 50)
        print(f"Coverage: {metrics['coverage']:.3f} (target: {metrics['target_coverage']:.3f})")
        print(f"Mean Width: {metrics['mean_width']:.3f}")
        print(f"Pinball Loss: {metrics['pinball_loss']:.3f}")

        # Display baseline comparisons
        if self.comparison_results.get("baseline_test_results"):
            print("\n🏆 Model vs Baseline Performance:")
            print("-" * 50)

            baseline_results = self.comparison_results["baseline_test_results"]
            for strategy_name, baseline_metrics in baseline_results.items():
                strategy_display = strategy_name.replace('baseline_', '').replace('_', ' ').title()
                print(f"\n{strategy_display}:")

                for metric_name in ["coverage", "mean_width", "pinball_loss"]:
                    if metric_name in metrics and metric_name in baseline_metrics:
                        trained_value = metrics[metric_name]
                        baseline_value = baseline_metrics[metric_name]

                        # Determine if improvement is good
                        if metric_name == "coverage":
                            trained_diff = abs(trained_value - self.config.evaluation.target_coverage)
                            baseline_diff = abs(baseline_value - self.config.evaluation.target_coverage)
                            improvement = baseline_diff - trained_diff
                        else:  # mean_width, pinball_loss (lower is better)
                            improvement = baseline_value - trained_value

                        improvement_sign = "✅" if improvement > 0 else "❌" if improvement < 0 else "➡️"
                        print(f"  {metric_name}: {trained_value:.4f} vs {baseline_value:.4f} {improvement_sign}")

        print(f"\n🔧 Model ready for inference: {self.production_predictor}")
        print("📊 All metrics and model artifacts logged to Wandb")


if __name__ == "__main__":
    TrainingFlow()
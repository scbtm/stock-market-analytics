"""
CatBoost Quantile Regression Training Flow

A Metaflow pipeline for training and evaluating quantile regression models
on stock market data. This flow orchestrates the complete modeling workflow
following the established steps and flow architecture pattern.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from metaflow import FlowSpec, step, Parameter

from stock_market_analytics.modeling import modeling_steps


class CatBoostQuantileRegressionFlow(FlowSpec):
    """
    A Metaflow pipeline for training CatBoost quantile regression models.

    This flow orchestrates the complete modeling process:
    1. Loads and prepares features data
    2. Creates train/validation/test splits
    3. Trains quantile regression model
    4. Evaluates model performance comprehensively
    5. Optionally applies conformal calibration
    6. Saves model artifacts and results

    The flow expects a BASE_DATA_PATH environment variable pointing to the data directory.
    """

    # Flow parameters
    test_size = Parameter(
        "test-size", help="Fraction of data for test set", default=0.2
    )

    validation_size = Parameter(
        "validation-size", help="Fraction of data for validation set", default=0.1
    )

    time_span_days = Parameter(
        "time-span-days", help="Number of recent days to use (0 for all)", default=0
    )

    enable_cross_validation = Parameter(
        "enable-cv", help="Whether to run cross-validation", default=False
    )

    enable_conformal_calibration = Parameter(
        "enable-calibration",
        help="Whether to apply conformal calibration",
        default=False,
    )

    create_baselines = Parameter(
        "create-baselines",
        help="Whether to create baseline model comparisons",
        default=True,
    )

    output_dir = Parameter(
        "output-dir",
        help="Output directory for artifacts (relative to BASE_DATA_PATH)",
        default="model_outputs",
    )

    @step
    def start(self) -> None:
        """
        Initialize the training flow.

        Validates environment variables and sets up the modeling run.
        Creates a unique run identifier for organizing outputs.
        """
        print("🚀 Starting CatBoost Quantile Regression Training Flow...")

        # Validate required environment variables
        if not os.environ.get("BASE_DATA_PATH"):
            raise ValueError("BASE_DATA_PATH environment variable is required")

        self.base_data_path = Path(os.environ["BASE_DATA_PATH"])
        print(f"📁 Data directory: {self.base_data_path}")

        # Create timestamped output directory
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = (
            self.base_data_path / self.output_dir / f"cbm_qr_{self.run_timestamp}"
        )
        print(f"💾 Output directory: {self.output_path}")

        # Initialize run info
        self.run_info = {
            "timestamp": self.run_timestamp,
            "parameters": {
                "test_size": self.test_size,
                "validation_size": self.validation_size,
                "time_span_days": self.time_span_days,
                "enable_cv": self.enable_cross_validation,
                "enable_calibration": self.enable_conformal_calibration,
                "create_baselines": self.create_baselines,
            },
        }

        self.next(self.load_data)

    @step
    def load_data(self) -> None:
        """
        Load and prepare data for modeling.

        Loads the features dataset and prepares it for modeling by:
        1. Loading features from parquet file
        2. Selecting target and feature columns from config
        3. Filtering by time span if specified
        4. Removing rows with missing targets

        Raises:
            FileNotFoundError: If features file is not found
            ValueError: If no valid data remains after preparation
        """
        print("📊 Loading and preparing modeling data...")

        # Load features data
        self.raw_data = modeling_steps.load_features_data(self.base_data_path)
        print(f"✅ Loaded raw data with shape: {self.raw_data.shape}")

        # Prepare modeling data
        time_span = self.time_span_days if self.time_span_days > 0 else None
        self.X, self.y = modeling_steps.prepare_modeling_data(
            self.raw_data, time_span_days=time_span
        )

        print(f"✅ Prepared features shape: {self.X.shape}")
        print(f"✅ Prepared target shape: {len(self.y)}")
        print(f"📈 Target statistics: mean={self.y.mean():.6f}, std={self.y.std():.6f}")

        # Store data info
        self.data_info = {
            "raw_shape": self.raw_data.shape,
            "features_shape": self.X.shape,
            "target_length": len(self.y),
            "target_mean": float(self.y.mean()),
            "target_std": float(self.y.std()),
            "target_min": float(self.y.min()),
            "target_max": float(self.y.max()),
        }

        self.next(self.split_data)

    @step
    def split_data(self) -> None:
        """
        Create train/validation/test splits for time series modeling.

        Uses time-series splitting to respect temporal order and avoid
        data leakage. Creates three splits:
        - Training set: for model fitting
        - Validation set: for hyperparameter tuning and early stopping
        - Test set: for final performance evaluation
        """
        print("🔄 Creating time series data splits...")

        # Create splits
        splits = modeling_steps.create_modeling_splits(
            self.X,
            self.y,
            test_size=self.test_size,
            validation_size=self.validation_size,
        )

        (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        ) = splits

        print(f"📊 Train set size: {len(self.y_train)}")
        print(f"📊 Validation set size: {len(self.y_val)}")
        print(f"📊 Test set size: {len(self.y_test)}")

        # Store split info
        self.split_info = {
            "train_size": len(self.y_train),
            "val_size": len(self.y_val),
            "test_size": len(self.y_test),
            "test_split_ratio": self.test_size,
            "val_split_ratio": self.validation_size,
        }

        self.next(self.train_model)

    @step
    def train_model(self) -> None:
        """
        Train the CatBoost quantile regression model.

        Fits a quantile regression model using the training data with
        validation set for early stopping. Uses configuration parameters
        from the config module for model hyperparameters.
        """
        print("🤖 Training CatBoost quantile regression model...")

        # Train model with validation
        self.model = modeling_steps.fit_quantile_model(
            self.X_train, self.y_train, X_val=self.X_val, y_val=self.y_val
        )

        print("✅ Model training completed successfully")

        # Get feature importance
        self.feature_importance = self.model.get_feature_importance()

        # Print top features
        sorted_features = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        print("🔍 Top 5 most important features:")
        for feat, importance in sorted_features[:5]:
            print(f"   {feat}: {importance:.4f}")

        self.next(self.evaluate_model)

    @step
    def evaluate_model(self) -> None:
        """
        Evaluate model performance on the test set.

        Performs comprehensive evaluation including:
        - Quantile regression metrics (pinball loss, coverage)
        - Financial metrics (directional accuracy, Sharpe ratio)
        - Backtesting metrics (returns, drawdown, risk measures)
        - Prediction interval analysis
        """
        print("📈 Evaluating model performance on test set...")

        # Comprehensive evaluation
        self.evaluation_results = modeling_steps.evaluate_model_performance(
            self.model, self.X_test, self.y_test
        )

        # Print key results
        quantile_metrics = self.evaluation_results.get("quantile_metrics", {})
        financial_metrics = self.evaluation_results.get("financial_metrics", {})
        backtest_metrics = self.evaluation_results.get("backtest_metrics", {})

        print("📊 === Key Evaluation Results ===")
        print(
            f"   Mean Pinball Loss: {quantile_metrics.get('mean_pinball_loss', 'N/A'):.6f}"
        )
        print(
            f"   Directional Accuracy: {financial_metrics.get('directional_accuracy', 'N/A'):.4f}"
        )
        print(
            f"   Sharpe Ratio (Pred): {financial_metrics.get('sharpe_ratio_pred', 'N/A'):.4f}"
        )
        print(f"   Total Return: {backtest_metrics.get('total_return', 'N/A'):.4f}")
        print(f"   Max Drawdown: {backtest_metrics.get('max_drawdown', 'N/A'):.4f}")

        # Decide next steps based on parameters
        if self.enable_conformal_calibration:
            self.next(self.apply_calibration)
        elif self.enable_cross_validation:
            self.next(self.run_cross_validation)
        elif self.create_baselines:
            self.next(self.create_baseline_comparison)
        else:
            self.next(self.save_artifacts)

    @step
    def apply_calibration(self) -> None:
        """
        Apply conformal calibration to the trained model.

        Uses the validation set to learn conformal prediction intervals
        that provide guaranteed coverage. This step is optional and only
        runs if enable_conformal_calibration is True.
        """
        print("🎯 Applying conformal calibration...")

        # Apply conformal calibration using validation set
        self.calibrator = modeling_steps.apply_conformal_calibration(
            self.model,
            self.X_val,
            self.y_val,
            alpha=0.1,  # 90% prediction intervals
        )

        print("✅ Conformal calibration completed")

        # Create calibrated production pipeline
        self.production_pipeline = modeling_steps.create_production_pipeline(
            self.model, calibrator=self.calibrator
        )

        # Continue with next steps
        if self.enable_cross_validation:
            self.next(self.run_cross_validation)
        elif self.create_baselines:
            self.next(self.create_baseline_comparison)
        else:
            self.next(self.save_artifacts)

    @step
    def run_cross_validation(self) -> None:
        """
        Run walk-forward cross-validation.

        Performs time series cross-validation to get more robust
        performance estimates. Uses expanding window approach
        suitable for financial time series data.
        """
        print("🔄 Running walk-forward cross-validation...")

        # Set up cross-validation parameters
        total_samples = len(self.y)
        initial_train_size = int(total_samples * 0.5)  # Use 50% for initial training
        test_size = int(total_samples * 0.05)  # 5% for each test fold
        step_size = test_size // 2  # 50% overlap between folds

        # Run cross-validation
        self.cv_results = modeling_steps.run_walk_forward_validation(
            self.X,
            self.y,
            initial_train_size=initial_train_size,
            test_size=test_size,
            step_size=step_size,
            max_folds=5,  # Limit to 5 folds for computational efficiency
        )

        print(f"✅ Completed cross-validation with {len(self.cv_results)} folds")

        # Summarize CV results
        if self.cv_results:
            # Calculate mean metrics across folds
            mean_pinball_loss = sum(
                fold.get("quantile_metrics", {}).get("mean_pinball_loss", 0)
                for fold in self.cv_results
            ) / len(self.cv_results)

            mean_directional_accuracy = sum(
                fold.get("financial_metrics", {}).get("directional_accuracy", 0)
                for fold in self.cv_results
            ) / len(self.cv_results)

            print("📊 === Cross-Validation Results (Mean) ===")
            print(f"   Mean Pinball Loss: {mean_pinball_loss:.6f}")
            print(f"   Directional Accuracy: {mean_directional_accuracy:.4f}")

        # Continue to next step
        if self.create_baselines:
            self.next(self.create_baseline_comparison)
        else:
            self.next(self.save_artifacts)

    @step
    def create_baseline_comparison(self) -> None:
        """
        Create baseline model comparisons.

        Fits simple baseline models (mean, median, last value, trend)
        for comparison with the main quantile regression model.
        Provides context for evaluating model performance.
        """
        print("📏 Creating baseline model comparisons...")

        self.baseline_results = modeling_steps.create_baseline_models(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        print("📊 === Baseline Model Results ===")
        for name, results in self.baseline_results.items():
            metrics = results["metrics"]
            dir_acc = metrics.get("directional_accuracy", 0)
            print(f"   {name.capitalize()}: Directional Accuracy = {dir_acc:.4f}")

        self.next(self.save_artifacts)

    @step
    def save_artifacts(self) -> None:
        """
        Save model artifacts and results.

        Persists the trained model, evaluation results, predictions,
        and configuration to the output directory. Creates a complete
        record of the modeling run for reproducibility.
        """
        print("💾 Saving model artifacts and results...")

        # Save main model and evaluation results
        self.artifact_paths = modeling_steps.save_model_artifacts(
            self.model,
            self.evaluation_results,
            self.output_path,
            model_name=f"catboost_qr_{self.run_timestamp}",
        )

        # Save test predictions
        test_predictions = self.model.predict(self.X_test)
        from stock_market_analytics.config import config

        predictions_path = modeling_steps.save_predictions(
            test_predictions,
            config.modeling.quantiles,
            self.output_path,
            filename=f"test_predictions_{self.run_timestamp}.parquet",
        )
        self.artifact_paths["predictions"] = predictions_path

        # Save configuration and run info
        config_dict = {
            "run_info": self.run_info,
            "data_info": self.data_info,
            "split_info": self.split_info,
            "quantiles": config.modeling.quantiles,
            "features": config.modeling.features,
            "target": config.modeling.target,
            "model_params": config.modeling.cb_model_params,
            "fit_params": config.modeling.cb_fit_params,
        }

        config_path = modeling_steps.save_configuration(
            config_dict,
            self.output_path,
            filename=f"run_config_{self.run_timestamp}.json",
        )
        self.artifact_paths["config"] = config_path

        # Save additional results if available
        if hasattr(self, "cv_results") and self.cv_results:
            import json

            cv_path = self.output_path / f"cv_results_{self.run_timestamp}.json"
            with open(cv_path, "w") as f:
                json.dump(self.cv_results, f, indent=2, default=str)
            self.artifact_paths["cv_results"] = cv_path

        if hasattr(self, "baseline_results") and self.baseline_results:
            import json

            baseline_path = (
                self.output_path / f"baseline_results_{self.run_timestamp}.json"
            )
            # Convert baseline results for JSON serialization
            baseline_serializable = {}
            for name, results in self.baseline_results.items():
                baseline_serializable[name] = {
                    "metrics": results["metrics"]
                    # Skip the actual model objects for serialization
                }

            with open(baseline_path, "w") as f:
                json.dump(baseline_serializable, f, indent=2, default=str)
            self.artifact_paths["baseline_results"] = baseline_path

        print("✅ All artifacts saved successfully:")
        for artifact_type, path in self.artifact_paths.items():
            print(f"   {artifact_type}: {path}")

        self.next(self.end)

    @step
    def end(self) -> None:
        """
        Final step of the training flow.

        Summarizes the completed modeling run and provides information
        about saved artifacts. Marks successful completion of the entire
        quantile regression modeling pipeline.
        """
        print("🎉 CatBoost Quantile Regression Training Flow completed successfully!")
        print(f"📁 Results saved to: {self.output_path}")
        print("📊 Model training and evaluation complete.")

        # Print final summary
        if hasattr(self, "evaluation_results"):
            quantile_metrics = self.evaluation_results.get("quantile_metrics", {})
            financial_metrics = self.evaluation_results.get("financial_metrics", {})

            print("\n📈 Final Model Performance:")
            print(
                f"   • Mean Pinball Loss: {quantile_metrics.get('mean_pinball_loss', 'N/A'):.6f}"
            )
            print(
                f"   • Directional Accuracy: {financial_metrics.get('directional_accuracy', 'N/A'):.4f}"
            )
            print(f"   • Model artifacts: {len(self.artifact_paths)} files saved")


if __name__ == "__main__":
    # Entry point for running the flow directly
    CatBoostQuantileRegressionFlow()

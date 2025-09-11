"""
CatBoost Quantile Regression Training Flow

A Metaflow pipeline for training and evaluating quantile regression models
on stock market data. This flow orchestrates the complete modeling workflow
following the established steps and flow architecture pattern.
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from metaflow import FlowSpec, step

from stock_market_analytics.modeling import modeling_steps
from stock_market_analytics.config import config

class TrainingFlow(FlowSpec):
    """
    A Metaflow pipeline for training CatBoost quantile regression models.

    This flow orchestrates the complete modeling process:
    1. Loads features
    2. Prepares training datasets
    3. Trains quantile regression model
    4. Evaluates model performance comprehensively
    5. Analyzes feature importance
    6. Applies optional conformal calibration
    7. Evaluates calibrated model

    The flow expects BASE_DATA_PATH and WANDB_KEY as environment variables pointing to the data directory and the Weights & Biases API Key respectively.
    """
    @step
    def start(self) -> None:
        """
        Initialize the training flow.

        Sets up configuration for the modeling run.
        """
        print("🚀 Starting CatBoost Quantile Regression Training Flow...")

        print("✅ Configuration loaded successfully")

        self.config = config
        self.modeling_config = config.modeling

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
        print(f"🗂️  Data path: {self.config.base_data_path}")

        # Load features data (error handling is in load_features_data)
        df = modeling_steps.load_features_data(self.config.base_data_path)
        print(f"✅ Loaded raw data with shape: {df.shape}")
        print(f"📈 Data date range: {df.index.min()} to {df.index.max()}")
        print(f"🔢 Number of features: {df.shape[1]}")
        print(f"🎯 Target column: {self.config.modeling.target}")

        self.df = df

        self.next(self.split_data)

    @step
    def split_data(self) -> None:
        """Split data into train/validation/calibration/test sets."""
        print("🔄 Splitting data into modeling sets...")
        
        self.modeling_sets = modeling_steps.prepare_modeling_data(self.df)
        
        # Print data split information
        for split_name, (X, y) in self.modeling_sets.items():
            print(f"📊 {split_name.upper()} set: {X.shape[0]} samples, {X.shape[1]} features")
            
        print("✅ Data splitting completed successfully")
        self.next(self.train_model)

    @step
    def train_model(self) -> None:
        """Train the CatBoost quantile regression model with early stopping."""
        print("🚀 Starting model training...")
        
        try:
            # Set up early stopping parameters
            fit_params = config.modeling.cb_fit_params.copy()
            print(f"📋 Training parameters: {list(fit_params.keys())}")

            # Get evaluation set
            xtrain, ytrain = self.modeling_sets["train"]
            xval, yval = self.modeling_sets["val"]
            print(f"🎯 Training on {len(ytrain)} samples, validating on {len(yval)} samples")

            print("🔧 Pre-fitting data transformation pipeline...")
            transformations = modeling_steps.get_adhoc_transforms_pipeline()
            xtrain, ytrain = self.modeling_sets["train"]

            # Fit transformations on training data only
            transformations.fit(xtrain)
            print("✅ Transformation pipeline fitted")

            _xtrain = transformations.transform(xtrain)
            _xval = transformations.transform(xval)
            print(f"🔄 Transformed training data shape: {_xtrain.shape}")
            print(f"🔄 Transformed validation data shape: {_xval.shape}")

            # Initialize model for early stopping
            print("🤖 Initializing CatBoost model...")
            model = modeling_steps.get_catboost_multiquantile_model()
            model_params = model.get_params()
            print(f"📊 Model quantiles: {model_params.get('quantiles', 'default')}")

            # Train the model to determine best iterations
            print("⏰ Training model with early stopping...")
            model.fit(_xtrain, ytrain, **fit_params | {"eval_set": (_xval, yval)})
            best_iteration = model.best_iteration_
            print(f"🎯 Best iteration found: {best_iteration}")

            model_params.update({"num_boost_round": best_iteration})
            self.model_params = model_params

            # Train final model on combined train+val
            print("🔄 Training final model on combined train+validation data...")
            transformations = modeling_steps.get_adhoc_transforms_pipeline()
            model = modeling_steps.get_catboost_multiquantile_model(model_params)

            xtrain_combined = pd.concat([xtrain, xval], axis=0)
            ytrain_combined = pd.concat([ytrain, yval], axis=0)
            print(f"📈 Combined training data shape: {xtrain_combined.shape}")

            pipeline = Pipeline(steps=[
                ("transformations", transformations),
                ("model", model),
            ])

            # Remove early stopping params for final fit
            fit_params.pop("early_stopping_rounds", None)
            
            # Prefix fit_params for pipeline model step
            pipeline_fit_params = {f"model__{k}": v for k, v in fit_params.items()}

            print("🏋️ Training final pipeline...")
            pipeline.fit(xtrain_combined, ytrain_combined, **pipeline_fit_params)
            print("✅ Model training completed successfully")

            self.pipeline = pipeline
            
        except Exception as e:
            print(f"❌ Error during model training: {e}")
            raise

        self.next(self.calibrate_model)

    @step
    def calibrate_model(self) -> None:
        """Apply conformal calibration to the trained model."""
        print("🎯 Starting model calibration...")
        
        try:
            modeling_sets = self.modeling_sets
            xcal, ycal = modeling_sets["cal"]
            print(f"📊 Calibration set size: {len(ycal)} samples")

            pipeline = self.pipeline

            print("🔮 Generating predictions on calibration set...")
            y_pred_cal = pipeline.predict(xcal)
            print(f"✅ Generated predictions shape: {y_pred_cal.shape}")

            print("⚙️ Initializing conformal calibrator...")
            conformal_calibrator = modeling_steps.get_calibrator()

            # Fit calibrator using calibration set
            print("🔧 Fitting conformal calibrator...")
            conformal_calibrator.fit(y_pred_cal=y_pred_cal, y_true_cal=ycal)

            # Create calibrated pipeline
            pipeline = Pipeline(steps=[
                ("transformations", pipeline.named_steps["transformations"]),
                ("model", pipeline.named_steps["model"]),
                ("calibrator", conformal_calibrator),
            ])

            print("✅ Conformal calibrator fitted successfully")
            print(f"📏 Learned radius (conformity score quantile): {conformal_calibrator.radius_:.4f}")
            
            self.pipeline = pipeline
            
        except Exception as e:
            print(f"❌ Error during model calibration: {e}")
            raise
            
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self) -> None:
        """Evaluate the calibrated model on test set."""
        print("📊 Starting model evaluation...")
        
        try:
            print("⚙️ Initializing evaluator...")
            evaluator = modeling_steps.get_evaluator()

            modeling_sets = self.modeling_sets
            xtest, ytest = modeling_sets["test"]
            print(f"🧪 Test set size: {len(ytest)} samples")

            # Generate prediction intervals for test set
            print("🔮 Generating prediction intervals on test set...")
            intervals_test = self.pipeline.predict(xtest)
            lower_bounds = intervals_test[:, 0]
            upper_bounds = intervals_test[:, 1]
            print(f"✅ Generated {len(intervals_test)} prediction intervals")
            
            # Print some sample predictions for transparency
            print(f"📈 Sample predictions (first 3):")
            for i in range(min(3, len(intervals_test))):
                print(f"   Sample {i+1}: [{lower_bounds[i]:.4f}, {upper_bounds[i]:.4f}] vs actual {ytest.iloc[i]:.4f}")

            # Evaluate the calibrated intervals
            print("📏 Evaluating prediction intervals (80% confidence)...")
            interval_metrics = evaluator.evaluate_intervals(
                y_true=ytest,
                y_lower=lower_bounds,
                y_upper=upper_bounds,
                alpha=0.2,  # 80% prediction intervals
            )

            print("\n📊 Calibrated Interval Evaluation Metrics:")
            print("="*50)
            for metric, value in interval_metrics.items():
                print(f"  📌 {metric}: {value:.4f}")
            print("="*50)
            
            # Store metrics for potential downstream use
            self.evaluation_metrics = interval_metrics
            
        except Exception as e:
            print(f"❌ Error during model evaluation: {e}")
            raise

        self.next(self.end)

    @step
    def end(self) -> None:
        """Finalize the training flow."""
        print("\n🏁 Training flow completed successfully!")
        print("\n📋 Flow Summary:")
        print("="*40)
        print("✅ Data loaded and processed")
        print("✅ Model trained with early stopping")
        print("✅ Conformal calibration applied")
        print("✅ Model evaluation completed")
        
        if hasattr(self, 'evaluation_metrics'):
            # Find the most relevant metric to highlight
            coverage_metrics = {k: v for k, v in self.evaluation_metrics.items() if 'coverage' in k.lower()}
            if coverage_metrics:
                best_metric = max(coverage_metrics.items(), key=lambda x: x[1])
                print(f"🎯 Key result: {best_metric[0]} = {best_metric[1]:.4f}")
        
        print("="*40)
        print("🚀 Ready for deployment or further analysis!")


if __name__ == "__main__":
    TrainingFlow()
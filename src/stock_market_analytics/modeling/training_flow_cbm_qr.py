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

        config.modeling.validate()
        print("✅ Configuration validated successfully")

        self.config = config.modeling

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
        df = modeling_steps.load_features_data(self.config.base_data_path)
        print(f"✅ Loaded raw data with shape: {df.shape}")

        self.df = df

        self.next(self.split_data)

    @step
    def split_data(self) -> None:
        self.modeling_sets = modeling_steps.create_modeling_splits(self.df)
        self.next(self.train_model)

    @step
    def train_model(self) -> None:
        # Set up early stopping parameters
        fit_params = config.modeling.cb_fit_params.copy()

        #get eval set
        xtrain, ytrain = self.modeling_sets["train"]
        xval, yval = self.modeling_sets["val"]

        print("🔧 Pre-fitting data transformation pipeline...")

        modeling_sets = self.modeling_sets

        transformations = modeling_steps.get_adhoc_transforms_pipeline()

        xtrain, _ = modeling_sets["train"]

        transformations.fit(xtrain)  # Fit PCA on training data only

        _xtrain = transformations.transform(xtrain)
        _xval = transformations.transform(xval)

        model = modeling_steps.get_catboost_multiquantile_model()

        model_params = model.get_params()

        # Train the pipeline to get best iterations
        model.fit(_xtrain, ytrain, **fit_params | {"eval_set": (_xval, yval)})
        best_iteration = model.best_iteration_

        model_params.update({"num_boost_round": best_iteration})

        self.model_params = model_params


        #Train final model on combined train+val
        transformations = modeling_steps.get_adhoc_transforms_pipeline()
        model = modeling_steps.get_catboost_multiquantile_model(model_params)

        xtrain = pd.concat([xtrain, xval], axis=0)
        ytrain = pd.concat([ytrain, yval], axis=0)

        pipeline = Pipeline(steps=[
            ("transformations", transformations),
            ("model", model),
        ])

        #Remove early stopping params for final fit
        fit_params.pop("early_stopping_rounds", None)

        pipeline.fit(xtrain, ytrain, **fit_params)

        self.pipeline = pipeline

        self.next(self.calibrate_model)

    @step
    def calibrate_model(self) -> None:

        modeling_sets = self.modeling_sets

        xcal, ycal = modeling_sets["cal"]

        pipeline = self.pipeline

        y_pred_cal = pipeline.predict(xcal)

        conformal_calibrator = modeling_steps.get_calibrator()

        # Fit calibrator using calibration set
        conformal_calibrator.fit(y_pred_cal=y_pred_cal, y_true_cal=ycal)

        pipeline = Pipeline(steps=[
            ("transformations", pipeline.named_steps["transformations"]),
            ("model", pipeline.named_steps["model"]),
            ("calibrator", conformal_calibrator),
        ])

        print(f"Conformal calibrator fitted")
        print(f"Learned radius (conformity score quantile): {conformal_calibrator.radius_:.4f}")
        self.pipeline = pipeline
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self) -> None:

        evaluator = modeling_steps.get_evaluator()

        modeling_sets = self.modeling_sets
        xtest, ytest = modeling_sets["test"]

        # Generate prediction intervals for test set
        intervals_test = self.pipeline.predict(xtest)
        lower_bounds = intervals_test[:, 0]
        upper_bounds = intervals_test[:, 1]

        # Evaluate the calibrated intervals
        interval_metrics = evaluator.evaluate_intervals(
            y_true=ytest,
            y_lower=lower_bounds,
            y_upper=upper_bounds,
            alpha=0.2,  # 80% prediction intervals
        )

        print("Calibrated Interval Evaluation Metrics:")
        for metric, value in interval_metrics.items():
            print(f"  {metric}: {value:.4f}")

        self.next(self.end)

    @step
    def end(self) -> None:
        # Finalize the training flow.
        print("🏁 Training flow completed successfully.")


if __name__ == "__main__":
    TrainingFlow()
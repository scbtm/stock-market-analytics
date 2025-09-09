"""
Rich protocols for the model factory components.

These protocols define the expected interfaces for all model factory components,
ensuring type safety and enforcing correct use of the modeling pipeline.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl


@runtime_checkable
class Calibrator(Protocol):
    """Protocol for model calibrators that perform post-hoc prediction processing."""

    def calibrate(
        self, X_cal: pl.DataFrame, y_cal: pl.Series, **kwargs: Any
    ) -> "Calibrator":
        """
        Learn calibration parameters from calibration data.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
            **kwargs: Additional calibration parameters

        Returns:
            Self for method chaining
        """
        ...

    def fit(self, X_cal: pl.DataFrame, y_cal: pl.Series) -> "Calibrator":
        """
        Sklearn-compatible fit method that calls calibrate.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets

        Returns:
            Self for method chaining
        """
        ...

    def predict(self, y_hat: np.ndarray) -> np.ndarray:
        """
        Apply calibration to model predictions.

        Args:
            y_hat: Raw model predictions

        Returns:
            Calibrated predictions
        """
        ...


@runtime_checkable
class QuantileCalibrator(Calibrator, Protocol):
    """Protocol for quantile-based calibrators."""

    def predict_quantiles(
        self, y_hat: np.ndarray, quantiles: list[float]
    ) -> np.ndarray:
        """
        Predict quantiles for given predictions.

        Args:
            y_hat: Raw model predictions
            quantiles: List of quantiles to predict

        Returns:
            Quantile predictions with shape (n_samples, n_quantiles)
        """
        ...


@runtime_checkable
class ProbabilityCalibrator(Calibrator, Protocol):
    """Protocol for probability calibrators."""

    def predict_proba(self, y_hat: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            y_hat: Raw model predictions

        Returns:
            Class probabilities
        """
        ...


@runtime_checkable
class DataSplitter(Protocol):
    """Protocol for data splitting strategies."""

    def split(
        self, X: pl.DataFrame, y: pl.Series
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        """
        Split data according to the splitting strategy.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        ...

    def get_train_indices(self, X: pl.DataFrame) -> list[int]:
        """
        Get indices for training data.

        Args:
            X: Feature matrix

        Returns:
            List of training indices
        """
        ...

    def get_test_indices(self, X: pl.DataFrame) -> list[int]:
        """
        Get indices for test data.

        Args:
            X: Feature matrix

        Returns:
            List of test indices
        """
        ...


@runtime_checkable
class PostProcessor(Protocol):
    """Protocol for applying business rules to predictions."""

    def apply_rules(
        self, predictions: np.ndarray, context: dict[str, Any]
    ) -> np.ndarray:
        """
        Apply business rules to model predictions.

        Args:
            predictions: Raw model predictions
            context: Additional context for rule application

        Returns:
            Post-processed predictions
        """
        ...

    def validate_predictions(self, predictions: np.ndarray) -> bool:
        """
        Validate predictions meet business constraints.

        Args:
            predictions: Model predictions to validate

        Returns:
            True if predictions are valid, False otherwise
        """
        ...


@runtime_checkable
class ModelEvaluator(Protocol):
    """Protocol for standardized model evaluation."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Evaluate model predictions against true values.

        Args:
            y_true: True target values
            y_pred: Model predictions

        Returns:
            Dictionary of evaluation metrics
        """
        ...

    def get_metric_names(self) -> list[str]:
        """
        Get names of metrics computed by this evaluator.

        Returns:
            List of metric names
        """
        ...


@runtime_checkable
class QuantileEvaluator(ModelEvaluator, Protocol):
    """Protocol for evaluating quantile predictions."""

    def evaluate_quantiles(
        self, y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: list[float]
    ) -> dict[str, float]:
        """
        Evaluate quantile predictions.

        Args:
            y_true: True target values
            y_pred_quantiles: Quantile predictions with shape (n_samples, n_quantiles)
            quantiles: List of quantiles corresponding to predictions

        Returns:
            Dictionary of quantile-specific metrics
        """
        ...


@runtime_checkable
class SklearnCompatibleEstimator(Protocol):
    """Protocol for sklearn-compatible estimators with additional functionality."""

    def fit(
        self, X: pl.DataFrame, y: pl.Series, **kwargs: Any
    ) -> "SklearnCompatibleEstimator":
        """
        Fit the estimator to training data.

        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional fit parameters

        Returns:
            Self for method chaining
        """
        ...

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features for prediction

        Returns:
            Predictions
        """
        ...

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        ...


@runtime_checkable
class QuantileEstimator(SklearnCompatibleEstimator, Protocol):
    """Protocol for quantile regression estimators."""

    def predict_quantiles(self, X: pl.DataFrame, quantiles: list[float]) -> np.ndarray:
        """
        Predict quantiles for given features.

        Args:
            X: Features for prediction
            quantiles: List of quantiles to predict

        Returns:
            Quantile predictions with shape (n_samples, n_quantiles)
        """
        ...


@runtime_checkable
class ModelingPipeline(Protocol):
    """Protocol for complete modeling pipelines."""

    def fit(self, X: pl.DataFrame, y: pl.Series) -> "ModelingPipeline":
        """
        Fit the entire modeling pipeline.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Self for method chaining
        """
        ...

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        Make predictions using the full pipeline.

        Args:
            X: Features for prediction

        Returns:
            Final pipeline predictions
        """
        ...

    def evaluate(self, X: pl.DataFrame, y: pl.Series) -> dict[str, float]:
        """
        Evaluate the pipeline on given data.

        Args:
            X: Features for evaluation
            y: True targets for evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        ...

    def get_components(self) -> dict[str, Any]:
        """
        Get all pipeline components.

        Returns:
            Dictionary mapping component names to component objects
        """
        ...

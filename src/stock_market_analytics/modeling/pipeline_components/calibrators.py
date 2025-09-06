import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from stock_market_analytics.config import config
from stock_market_analytics.modeling.pipeline_components.functions import (
    apply_conformal,
    conformal_adjustment,
)


class ConformalCalibrator(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible conformal calibrator for quantile regression models.

    This calibrator implements Conformalized Quantile Regression (CQR) and can be
    used as the final step in a scikit-learn pipeline. It stores calibration data
    during fit() and applies conformal adjustments during predict().

    The calibrator returns only the conformally adjusted bounds (lower, upper)
    when predict() is called, making it suitable for direct use in production.
    """

    def __init__(
        self,
        target_coverage: float = None,
        low_idx: int = None,
        high_idx: int = None,
        store_calibration_data: bool = True,
    ):
        """
        Initialize the conformal calibrator.

        Args:
            target_coverage: Target coverage level (e.g., 0.8 for 80% coverage)
                           Defaults to config.modeling.target_coverage
            low_idx: Index of the lower quantile in model predictions
                    Defaults to config.modeling.quantile_indices["LOW"]
            high_idx: Index of the upper quantile in model predictions
                     Defaults to config.modeling.quantile_indices["HIGH"]
            store_calibration_data: Whether to store calibration predictions
                                  Set to False to save memory if not needed
        """
        self.target_coverage = target_coverage or config.modeling.target_coverage
        self.low_idx = low_idx if low_idx is not None else config.modeling.quantile_indices["LOW"]
        self.high_idx = high_idx if high_idx is not None else config.modeling.quantile_indices["HIGH"]
        self.store_calibration_data = store_calibration_data

        # Will be set during fit()
        self.conformal_quantile_ = None
        self.alpha_ = 1 - self.target_coverage
        self.is_fitted_ = False

        # Optional storage for debugging/analysis
        self.calibration_predictions_ = None
        self.calibration_targets_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray):
        """
        Fit the conformal calibrator using calibration data.

        This method expects that X contains predictions from the upstream model
        (i.e., quantile predictions with shape (n_samples, n_quantiles)).
        The calibrator will compute the conformal quantile based on these predictions
        and the true targets y.

        Args:
            X: Model predictions on calibration set, shape (n_samples, n_quantiles)
            y: True targets for calibration set, shape (n_samples,) or (n_samples, 1)

        Returns:
            self: Returns self for method chaining
        """
        # Convert inputs to numpy arrays
        X_array = np.asarray(X)
        if hasattr(y, "values"):
            y_array = y.values.ravel()
        else:
            y_array = np.asarray(y).ravel()

        # Validate inputs
        if X_array.ndim != 2:
            raise ValueError(
                f"X must be 2D array with shape (n_samples, n_quantiles), got shape {X_array.shape}"
            )

        if len(y_array) != X_array.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. X: {X_array.shape[0]}, y: {len(y_array)}"
            )

        if self.low_idx >= X_array.shape[1] or self.high_idx >= X_array.shape[1]:
            raise ValueError(
                f"Quantile indices out of bounds. X has {X_array.shape[1]} quantiles, "
                f"but low_idx={self.low_idx}, high_idx={self.high_idx}"
            )

        # Extract the relevant quantiles for conformal adjustment
        q_lo_cal = X_array[:, self.low_idx]
        q_hi_cal = X_array[:, self.high_idx]

        # Compute conformal quantile
        self.conformal_quantile_ = conformal_adjustment(
            q_lo_cal, q_hi_cal, y_array, alpha=self.alpha_
        )

        # Optionally store calibration data for analysis
        if self.store_calibration_data:
            self.calibration_predictions_ = X_array.copy()
            self.calibration_targets_ = y_array.copy()

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Apply conformal adjustment to predictions.

        Args:
            X: Model predictions, shape (n_samples, n_quantiles)

        Returns:
            Conformally adjusted bounds, shape (n_samples, 2)
            Column 0: Lower bound, Column 1: Upper bound
        """
        if not self.is_fitted_:
            raise ValueError(
                "Calibrator must be fitted before transform. Call fit() first."
            )

        X_array = np.asarray(X)

        if X_array.ndim != 2:
            raise ValueError(
                f"X must be 2D array with shape (n_samples, n_quantiles), got shape {X_array.shape}"
            )

        if self.low_idx >= X_array.shape[1] or self.high_idx >= X_array.shape[1]:
            raise ValueError(
                f"Quantile indices out of bounds. X has {X_array.shape[1]} quantiles, "
                f"but low_idx={self.low_idx}, high_idx={self.high_idx}"
            )

        # Extract the relevant quantiles
        q_lo = X_array[:, self.low_idx]
        q_hi = X_array[:, self.high_idx]

        # Apply conformal adjustment
        lo_conformal, hi_conformal = apply_conformal(
            q_lo, q_hi, self.conformal_quantile_
        )

        # Return as 2-column array: [lower_bounds, upper_bounds]
        return np.column_stack([lo_conformal, hi_conformal])

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict conformally adjusted bounds.

        This method is an alias for transform() to provide a more intuitive
        interface when the calibrator is used as the final step in a pipeline.

        Args:
            X: Model predictions, shape (n_samples, n_quantiles)

        Returns:
            Conformally adjusted bounds, shape (n_samples, 2)
            Column 0: Lower bound, Column 1: Upper bound
        """
        return self.transform(X)

    def get_conformal_info(self) -> dict:
        """
        Get information about the fitted conformal calibrator.

        Returns:
            Dictionary containing calibrator information
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted first.")

        info = {
            "target_coverage": self.target_coverage,
            "alpha": self.alpha_,
            "conformal_quantile": self.conformal_quantile_,
            "low_idx": self.low_idx,
            "high_idx": self.high_idx,
            "is_fitted": self.is_fitted_,
        }

        if self.store_calibration_data and self.calibration_predictions_ is not None:
            info.update(
                {
                    "calibration_samples": len(self.calibration_targets_),
                    "calibration_quantiles": self.calibration_predictions_.shape[1],
                }
            )

        return info


class PipelineWithCalibrator:
    """
    Helper class to create and manage pipelines with conformal calibration.

    This class provides utilities to add a fitted calibrator to an existing pipeline
    and manage the two-stage fitting process (model first, then calibrator).
    """

    @staticmethod
    def add_calibrator_to_pipeline(
        base_pipeline: Pipeline,
        calibrator: ConformalCalibrator,
        calibrator_name: str = "conformal_calibrator",
    ) -> Pipeline:
        """
        Add a fitted calibrator as the final step of an existing pipeline.

        Args:
            base_pipeline: The base pipeline (should be already fitted)
            calibrator: A fitted ConformalCalibrator instance
            calibrator_name: Name for the calibrator step

        Returns:
            New pipeline with calibrator added
        """
        if not calibrator.is_fitted_:
            raise ValueError("Calibrator must be fitted before adding to pipeline.")

        # Create new pipeline steps
        new_steps = base_pipeline.steps + [(calibrator_name, calibrator)]

        # Create new pipeline
        calibrated_pipeline = Pipeline(new_steps)

        # Copy fitted state from base pipeline
        for step_name, step_transformer in base_pipeline.named_steps.items():
            calibrated_pipeline.named_steps[step_name] = step_transformer

        return calibrated_pipeline

    @staticmethod
    def create_calibrated_pipeline(
        base_pipeline: Pipeline,
        X_cal: pd.DataFrame | np.ndarray,
        y_cal: pd.DataFrame | np.ndarray,
        calibrator_params: dict = None,
    ) -> tuple[Pipeline, ConformalCalibrator]:
        """
        Create a calibrated pipeline by fitting a calibrator on calibration data.

        Args:
            base_pipeline: Fitted base pipeline that produces quantile predictions
            X_cal: Calibration features
            y_cal: Calibration targets
            calibrator_params: Parameters for ConformalCalibrator initialization

        Returns:
            Tuple of (calibrated_pipeline, fitted_calibrator)
        """
        calibrator_params = calibrator_params or {}

        # Create and fit calibrator
        calibrator = ConformalCalibrator(**calibrator_params)

        # Get predictions from base pipeline for calibration
        cal_predictions = base_pipeline.predict(X_cal)

        # Fit calibrator
        calibrator.fit(cal_predictions, y_cal)

        # Add calibrator to pipeline
        calibrated_pipeline = PipelineWithCalibrator.add_calibrator_to_pipeline(
            base_pipeline, calibrator
        )

        return calibrated_pipeline, calibrator

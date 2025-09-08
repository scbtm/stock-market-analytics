"""
Scikit-learn compatible model implementations.

Clean, focused implementations that follow our simplified protocol.
"""

from typing import Any, Sequence
import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y

from ..protocols import QuantilePredictor
from .prediction_functions import create_catboost_pool


class CatBoostMultiQuantileModel(BaseEstimator, RegressorMixin, QuantilePredictor):
    """Scikit-learn compatible wrapper for CatBoost multi-quantile regression."""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        random_state: int = 1,
        verbose: bool = False,
        **catboost_params: Any,
    ):
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.random_state = random_state
        self.verbose = verbose
        self.catboost_params = catboost_params
        self._model = None
        
        # Calibration state - calibrator does the work, we just hold reference
        self._calibrator = None

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "CatBoostMultiQuantileModel":
        """Fit the CatBoost multi-quantile model."""
        X, y = check_X_y(X, y, accept_sparse=False)

        # Build CatBoost parameters
        params = self.catboost_params.copy()
        
        # Set up multi-quantile loss
        alpha_str = ",".join([str(q) for q in sorted(self.quantiles)])
        params["loss_function"] = f"MultiQuantile:alpha={alpha_str}"
        params["random_state"] = self.random_state
        params["verbose"] = self.verbose

        # Create and fit model
        self._model = CatBoostRegressor(**params)
        train_pool = create_catboost_pool(X, y)
        self._model.fit(train_pool, **fit_params)

        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()

        return self

    def predict_quantiles(self, X: Any, quantiles: Sequence[float]) -> np.ndarray:
        """Generate multi-quantile predictions (calibrated if calibrator is fitted)."""
        if self._model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Get raw predictions from CatBoost - inline the simple logic
        # Note: CatBoost MultiQuantile ignores the quantiles parameter 
        # and returns predictions for all trained quantiles
        pool = create_catboost_pool(X)
        raw_predictions = np.asarray(self._model.predict(pool))
        
        # Ensure proper shape
        if raw_predictions.ndim == 1:
            raw_predictions = raw_predictions.reshape(-1, 1)
        
        # Enforce non-crossing quantiles
        raw_predictions.sort(axis=1)
        
        # Return raw predictions - calibration only applies to intervals
        # Individual quantiles don't get "calibrated" in conformal prediction
        return raw_predictions

    def predict(self, X: Any) -> np.ndarray:
        """Generate point predictions (median quantile)."""
        quantile_preds = self.predict_quantiles(X, [0.5])
        # Find median quantile (0.5)
        median_idx = None
        for i, q in enumerate(self.quantiles):
            if abs(q - 0.5) < 1e-6:  # Close to 0.5
                median_idx = i
                break
        
        if median_idx is None:
            # If no exact 0.5 quantile, use middle one
            median_idx = len(self.quantiles) // 2
            
        return quantile_preds[:, median_idx]
    
    def calibrate(self, X_cal: Any, y_cal: Any, calibrator) -> None:
        """
        Calibrate the model using the provided calibrator.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets  
            calibrator: Calibrator instance with fit method (e.g., ConformalCalibrator)
        """
        if self._model is None:
            raise ValueError("Model must be fitted before calibration")
        
        # Get quantile predictions on calibration set
        q_cal = self.predict_quantiles(X_cal, self.quantiles)
        
        # Fit the calibrator - it does all the work
        calibrator.fit(y_cal, q_cal, self.quantiles)
        
        # Store reference to fitted calibrator
        self._calibrator = calibrator
        
        print(f"Model calibrated with {type(calibrator).__name__}")
    
    def predict_intervals(self, X: Any) -> np.ndarray:
        """
        Predict calibrated intervals.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, 2) with [lower, upper] bounds
        """
        if self._calibrator is None or not self._calibrator.is_fitted:
            raise ValueError("Model must be calibrated before predicting intervals. Call calibrate() first.")
        
        # Get raw quantile predictions - inline simple logic
        pool = create_catboost_pool(X)
        raw_predictions = np.asarray(self._model.predict(pool))
        
        # Ensure proper shape
        if raw_predictions.ndim == 1:
            raw_predictions = raw_predictions.reshape(-1, 1)
        
        # Enforce non-crossing quantiles
        raw_predictions.sort(axis=1)
        
        # Let calibrator generate intervals
        return self._calibrator.predict_intervals(raw_predictions, self.quantiles)
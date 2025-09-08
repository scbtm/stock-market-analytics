"""
Protocol-compliant calibrators for uncertainty quantification.

This module provides calibrators that implement the CalibratorProtocol,
enabling consistent interfaces for conformal prediction and other calibration methods.
"""

from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from stock_market_analytics.config import config
from stock_market_analytics.modeling.model_factory.calibration.calibration_functions import (
    apply_conformal,
    conformal_adjustment,
)
from stock_market_analytics.modeling.model_factory.protocols import (
    CalibratorProtocol,
    CalibrationKind,
    TaskType,
    PredKind,
    PredictionExtractor,
    GetSetParamsMixin,
    is_quantiles,
)
# Note: extract_quantiles was removed as it was over-abstracted
# Use model.predict_quantiles() directly instead

NDArrayF = npt.NDArray[np.float64]


class QuantileIntervalCalibrator(CalibratorProtocol):
    """
    Protocol-compliant conformal calibrator for quantile regression.
    
    This calibrator implements Conformalized Quantile Regression (CQR) and produces
    interval predictions with statistical coverage guarantees.
    """
    
    calibration_kind: CalibrationKind = "quantile_interval"
    task_type: TaskType = "quantile_regression"
    input_kind: tuple[PredKind, ...] = ("quantiles",)
    output_kind: tuple[PredKind, ...] = ("interval",)
    
    def __init__(
        self,
        target_coverage: float | None = None,
        low_quantile: float = 0.1,
        high_quantile: float = 0.9,
    ):
        """
        Initialize the quantile interval calibrator.
        
        Args:
            target_coverage: Target coverage level (e.g., 0.8 for 80% coverage)
            low_quantile: Lower quantile for interval construction
            high_quantile: Upper quantile for interval construction
        """
        self.target_coverage = target_coverage or config.modeling.target_coverage
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.alpha_ = 1 - self.target_coverage
        self.conformal_quantile_ = None
        self.is_fitted_ = False
    
    def fit(
        self,
        base_estimator: BaseEstimator,
        X_cal: Any,
        y_cal: Any,
        *,
        extract: PredictionExtractor | None = None,
        **kwargs: Any,
    ) -> "QuantileIntervalCalibrator":
        """
        Fit the calibrator using calibration data.
        
        Args:
            base_estimator: Fitted base estimator that can predict quantiles
            X_cal: Calibration features
            y_cal: Calibration targets
            extract: Optional custom extractor function
            **kwargs: Additional parameters
            
        Returns:
            self: Fitted calibrator
        """
        # Extract predictions using protocol-compliant approach
        if extract is None:
            # Use default quantile extraction with model's quantiles
            if hasattr(base_estimator, 'quantiles'):
                quantiles = base_estimator.quantiles
                preds = extract_quantiles(base_estimator, X_cal, quantiles)
            else:
                preds = extract_quantiles(base_estimator, X_cal)
        else:
            preds = extract(base_estimator, X_cal)
        
        if not is_quantiles(preds):
            raise ValueError(
                f"Calibrator expects quantile predictions, got {preds.kind}"
            )
        
        # Convert targets to array
        y_array = np.asarray(y_cal, dtype=float).ravel()
        
        # Find indices for low and high quantiles
        quantile_list = list(preds.quantiles)
        try:
            low_idx = quantile_list.index(self.low_quantile)
            high_idx = quantile_list.index(self.high_quantile)
        except ValueError as e:
            raise ValueError(
                f"Required quantiles {self.low_quantile}, {self.high_quantile} "
                f"not found in model predictions {quantile_list}"
            ) from e
        
        # Extract quantile predictions for calibration
        q_lo_cal = preds.q[:, low_idx]
        q_hi_cal = preds.q[:, high_idx]
        
        # Compute conformal quantile
        self.conformal_quantile_ = conformal_adjustment(
            q_lo_cal, q_hi_cal, y_array, alpha=self.alpha_
        )
        
        self.is_fitted_ = True
        return self
    
    def wrap(self, base_estimator: BaseEstimator) -> BaseEstimator:
        """
        Return a sklearn-compatible estimator that performs calibrated predictions.
        
        Returns:
            CalibratedQuantileWrapper: Wrapper that exposes predict_interval method
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted before wrapping.")
        
        return CalibratedQuantileWrapper(
            base_estimator=base_estimator,
            conformal_quantile=self.conformal_quantile_,
            low_quantile=self.low_quantile,
            high_quantile=self.high_quantile,
            target_coverage=self.target_coverage,
        )
    
    def info(self) -> Mapping[str, Any]:
        """Get information about the fitted calibrator."""
        info = {
            "calibration_kind": self.calibration_kind,
            "task_type": self.task_type,
            "target_coverage": self.target_coverage,
            "low_quantile": self.low_quantile,
            "high_quantile": self.high_quantile,
            "alpha": self.alpha_,
            "is_fitted": self.is_fitted_,
        }
        
        if self.is_fitted_:
            info["conformal_quantile"] = self.conformal_quantile_
        
        return info


class CalibratedQuantileWrapper(BaseEstimator, GetSetParamsMixin):
    """
    Sklearn-compatible wrapper for calibrated quantile predictions.
    
    This wrapper maintains the original model's quantile prediction capability
    while adding calibrated interval predictions.
    """
    
    def __init__(
        self,
        base_estimator: BaseEstimator,
        conformal_quantile: float,
        low_quantile: float,
        high_quantile: float,
        target_coverage: float,
    ):
        self.base_estimator = base_estimator
        self.conformal_quantile = conformal_quantile
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.target_coverage = target_coverage
    
    def predict(self, X: Any) -> NDArrayF:
        """Predict calibrated intervals (lo, hi)."""
        return self.predict_interval(X)
    
    def predict_quantiles(
        self, 
        X: Any, 
        quantiles: Any | None = None
    ) -> NDArrayF:
        """Pass through to base estimator's quantile predictions."""
        if hasattr(self.base_estimator, 'predict_quantiles'):
            return self.base_estimator.predict_quantiles(X, quantiles)
        else:
            # Fallback for models that don't implement the protocol
            return self.base_estimator.predict(X)
    
    def predict_interval(
        self, 
        X: Any, 
        coverage: float | None = None
    ) -> NDArrayF:
        """
        Predict calibrated intervals with guaranteed coverage.
        
        Args:
            X: Input features
            coverage: Coverage level (uses fitted coverage if None)
            
        Returns:
            Array of shape (n_samples, 2) with [lo, hi] intervals
        """
        if coverage is not None and coverage != self.target_coverage:
            raise ValueError(
                f"Coverage {coverage} differs from fitted coverage {self.target_coverage}. "
                "Re-fit calibrator with different target coverage."
            )
        
        # Get base quantile predictions
        q_preds = self.predict_quantiles(X)
        
        # Find quantile indices
        if hasattr(self.base_estimator, 'quantiles'):
            quantiles = list(self.base_estimator.quantiles)
            low_idx = quantiles.index(self.low_quantile)
            high_idx = quantiles.index(self.high_quantile)
        else:
            # Assume standard quantile ordering for backward compatibility
            low_idx = 0  # First quantile
            high_idx = -1  # Last quantile
        
        q_lo = q_preds[:, low_idx]
        q_hi = q_preds[:, high_idx]
        
        # Apply conformal adjustment
        lo_conformal, hi_conformal = apply_conformal(
            q_lo, q_hi, self.conformal_quantile
        )
        
        return np.column_stack([lo_conformal, hi_conformal])
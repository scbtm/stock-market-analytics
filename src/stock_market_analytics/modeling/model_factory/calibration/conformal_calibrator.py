"""
Conformal prediction calibrator for quantile regression models.

Implements Conformalized Quantile Regression (CQR) to provide statistical coverage guarantees.
"""

from typing import Any
import numpy as np
import numpy.typing as npt
from .calibration_functions import conformal_adjustment, apply_conformal

NDArrayF = npt.NDArray[np.float64]


class ConformalCalibrator:
    """
    Conformal prediction calibrator for quantile regression.
    
    This calibrator learns to adjust quantile predictions to achieve target coverage
    using conformal prediction theory.
    """
    
    def __init__(self, target_coverage: float = 0.8, low_quantile: float = 0.1, high_quantile: float = 0.9):
        """
        Initialize the conformal calibrator.
        
        Args:
            target_coverage: Target coverage level (e.g., 0.8 for 80% coverage)
            low_quantile: Lower quantile for interval construction
            high_quantile: Higher quantile for interval construction
        """
        self.target_coverage = target_coverage
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        
        # Fitted state
        self._is_fitted = False
        self._conformal_quantile = None
        self._alpha = 1 - target_coverage
    
    def fit(self, y_true: NDArrayF, y_pred_quantiles: NDArrayF, quantiles: list[float]) -> "ConformalCalibrator":
        """
        Fit the calibrator using calibration data.
        
        Args:
            y_true: True target values on calibration set
            y_pred_quantiles: Predicted quantiles on calibration set, shape (n_samples, n_quantiles)
            quantiles: List of quantile levels corresponding to y_pred_quantiles columns
            
        Returns:
            self: Fitted calibrator
        """
        # Find indices for required quantiles
        try:
            low_idx = quantiles.index(self.low_quantile)
            high_idx = quantiles.index(self.high_quantile)
        except ValueError:
            raise ValueError(
                f"Required quantiles {self.low_quantile}, {self.high_quantile} "
                f"not found in provided quantiles {quantiles}. "
                f"Model must be trained with these quantiles."
            )
        
        # Extract quantile predictions for calibration
        q_lo_cal = y_pred_quantiles[:, low_idx]
        q_hi_cal = y_pred_quantiles[:, high_idx]
        y_array = np.asarray(y_true).ravel()
        
        # Compute conformal adjustment
        self._conformal_quantile = conformal_adjustment(q_lo_cal, q_hi_cal, y_array, self._alpha)
        self._is_fitted = True
        
        return self
    
    # apply_calibration method removed - it was redundant with predict_intervals
    # Conformal prediction is fundamentally about intervals, not adjusting all quantiles
    
    def predict_intervals(self, y_pred_quantiles: NDArrayF, quantiles: list[float]) -> NDArrayF:
        """
        Get calibrated prediction intervals.
        
        Args:
            y_pred_quantiles: Raw quantile predictions
            quantiles: List of quantile levels
            
        Returns:
            Array of shape (n_samples, 2) with [lower, upper] bounds
        """
        if not self._is_fitted:
            raise ValueError("Calibrator must be fitted before predicting intervals")
        
        # Find indices for calibration quantiles
        try:
            low_idx = quantiles.index(self.low_quantile)
            high_idx = quantiles.index(self.high_quantile)
        except ValueError:
            raise ValueError(
                f"Calibration quantiles {self.low_quantile}, {self.high_quantile} "
                f"not found in prediction quantiles {quantiles}"
            )
        
        q_lo = y_pred_quantiles[:, low_idx]
        q_hi = y_pred_quantiles[:, high_idx]
        
        # Apply conformal adjustment
        lo_conformal, hi_conformal = apply_conformal(q_lo, q_hi, self._conformal_quantile)
        
        return np.column_stack([lo_conformal, hi_conformal])
    
    @property
    def is_fitted(self) -> bool:
        """Check if calibrator is fitted."""
        return self._is_fitted
    
    def get_info(self) -> dict[str, Any]:
        """Get information about the calibrator."""
        info = {
            "target_coverage": self.target_coverage,
            "low_quantile": self.low_quantile,
            "high_quantile": self.high_quantile,
            "is_fitted": self._is_fitted,
        }
        
        if self._is_fitted:
            info["conformal_quantile"] = self._conformal_quantile
        
        return info
"""
Core calibration functions for uncertainty quantification.

This module contains conformal prediction algorithms and other calibration methods
used by calibrators to provide statistical coverage guarantees.
"""

import numpy as np


def conformal_adjustment(
    q_lo_cal: np.ndarray, q_hi_cal: np.ndarray, y_cal: np.ndarray, alpha: float
) -> float:
    """
    Compute conformal adjustment for Conformalized Quantile Regression (CQR).
    
    Computes nonconformity scores s_i = max(q_lo - y, y - q_hi).
    Returns the (1-alpha) empirical quantile with finite-sample correction.
    
    Args:
        q_lo_cal: Lower quantile predictions on calibration set
        q_hi_cal: Higher quantile predictions on calibration set  
        y_cal: True values on calibration set
        alpha: Miscoverage level (e.g., 0.2 for 80% coverage)
        
    Returns:
        Conformal adjustment value to be added/subtracted from intervals
    """
    s = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    s_sorted = np.sort(s)
    n = len(s_sorted)
    # Finite-sample index per CQR (Romano et al.): ceil((n+1)*(1-alpha)) - 1
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = np.clip(k, 0, n - 1)

    return float(s_sorted[k])


def apply_conformal(
    q_lo: np.ndarray, q_hi: np.ndarray, q_conformal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply conformal adjustment to quantile predictions.
    
    Args:
        q_lo: Lower quantile predictions
        q_hi: Higher quantile predictions
        q_conformal: Conformal adjustment value (scalar or array)
        
    Returns:
        Tuple of (adjusted_lower, adjusted_upper) predictions
    """
    lo = q_lo - q_conformal
    hi = q_hi + q_conformal
    return lo, hi
"""
Calibration components for quantile regression models.

This module provides calibrators that can be used with fitted models to improve
their uncertainty quantification through post-hoc calibration methods.
"""

from .conformal_calibrator import ConformalCalibrator

__all__ = [
    "ConformalCalibrator",
]
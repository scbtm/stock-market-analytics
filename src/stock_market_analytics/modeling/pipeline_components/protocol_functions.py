"""
Simple utility functions for pipeline components.

Only the essentials - no over-abstracted extraction helpers.
"""

import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]

def validate_quantiles(quantiles: list[float]) -> None:
    """Basic validation for quantile values."""
    if not all(0 <= q <= 1 for q in quantiles):
        raise ValueError("Quantiles must be between 0 and 1")
    if quantiles != sorted(quantiles):
        raise ValueError("Quantiles must be sorted")

def compute_quantile_loss(y_true: NDArrayF, y_pred: NDArrayF, quantile: float) -> float:
    """Compute quantile loss for a single quantile."""
    residual = y_true - y_pred
    return np.mean(np.where(residual >= 0, quantile * residual, (quantile - 1) * residual))
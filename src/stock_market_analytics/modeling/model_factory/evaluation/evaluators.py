"""
Simple evaluator for quantile regression models.

Focused on what you actually need for experimentation.
"""

from typing import Sequence
import numpy as np
import numpy.typing as npt
from ..protocols import Evaluator
from .evaluation_functions import compute_quantile_loss

NDArrayF = npt.NDArray[np.float64]


class QuantileRegressionEvaluator(Evaluator):
    """Simple evaluator for quantile regression experiments."""
    
    def evaluate(self, y_true: NDArrayF, y_pred: NDArrayF, quantiles: Sequence[float]) -> dict[str, float]:
        """
        Evaluate quantile predictions with standard metrics.
        
        Args:
            y_true: Ground truth values (n_samples,)
            y_pred: Predicted quantiles (n_samples, n_quantiles)
            quantiles: Quantile levels corresponding to y_pred columns
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred)
        
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
            
        metrics = {}
        
        # Compute quantile loss for each quantile
        for i, q in enumerate(quantiles):
            loss = compute_quantile_loss(y_true, y_pred[:, i], q)
            metrics[f"quantile_loss_{q}"] = loss
        
        # Mean quantile loss
        all_losses = [metrics[f"quantile_loss_{q}"] for q in quantiles]
        metrics["mean_quantile_loss"] = np.mean(all_losses)
        
        # Coverage for interval (if we have quantiles that can form an interval)
        if len(quantiles) >= 2:
            # Find symmetric quantiles around 0.5
            low_q_idx = None
            high_q_idx = None
            for i, q in enumerate(quantiles):
                if q < 0.5:
                    low_q_idx = i
                elif q > 0.5 and high_q_idx is None:
                    high_q_idx = i
                    break
            
            if low_q_idx is not None and high_q_idx is not None:
                low_pred = y_pred[:, low_q_idx]
                high_pred = y_pred[:, high_q_idx]
                coverage = np.mean((y_true >= low_pred) & (y_true <= high_pred))
                mean_width = np.mean(high_pred - low_pred)
                
                metrics["interval_coverage"] = coverage
                metrics["mean_interval_width"] = mean_width
        
        return metrics
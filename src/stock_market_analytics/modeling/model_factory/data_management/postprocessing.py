"""
Post-processors for applying business rules to model predictions.

This module contains classes that apply domain-specific business rules
and constraints to raw model predictions.
"""

from typing import Any
import numpy as np
from stock_market_analytics.modeling.model_factory.protocols import PostProcessor, Array


class StockReturnPostProcessor(PostProcessor):
    """Post-processor for stock return predictions with business rule validation."""
    
    def __init__(
        self, 
        min_return: float = -0.5, 
        max_return: float = 0.5,
        clip_outliers: bool = True
    ):
        """Initialize with return bounds and processing options.
        
        Args:
            min_return: Minimum allowed return (e.g., -50%)
            max_return: Maximum allowed return (e.g., +50%)
            clip_outliers: Whether to clip values outside bounds
        """
        self.min_return = min_return
        self.max_return = max_return
        self.clip_outliers = clip_outliers
        
    def apply_rules(
        self,
        predictions: Array,
        context: dict[str, Any] | None = None,
    ) -> Array:
        """Apply business rules to predictions."""
        predictions = np.asarray(predictions, dtype=float)
        
        if self.clip_outliers:
            predictions = np.clip(predictions, self.min_return, self.max_return)
            
        # Replace any remaining NaN/inf values
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=self.max_return, neginf=self.min_return)
        
        return predictions
        
    def validate_predictions(self, predictions: Array) -> bool:
        """Return True if predictions satisfy business constraints."""
        predictions = np.asarray(predictions)
        
        # Check for NaN/inf values
        if not np.all(np.isfinite(predictions)):
            return False
            
        # Check bounds
        if np.any(predictions < self.min_return) or np.any(predictions > self.max_return):
            return False
            
        return True
        
    def report_violations(self, predictions: Array) -> dict[str, Any]:
        """Return a structured report of constraint violations."""
        predictions = np.asarray(predictions)
        violations = {}
        
        # Count NaN/inf values
        nan_count = np.sum(~np.isfinite(predictions))
        if nan_count > 0:
            violations['nan_inf_count'] = int(nan_count)
            
        # Count bound violations
        below_min = np.sum(predictions < self.min_return)
        above_max = np.sum(predictions > self.max_return)
        
        if below_min > 0:
            violations['below_min_count'] = int(below_min)
            violations['min_violation_value'] = float(np.min(predictions))
            
        if above_max > 0:
            violations['above_max_count'] = int(above_max)
            violations['max_violation_value'] = float(np.max(predictions))
            
        violations['total_samples'] = len(predictions)
        violations['violation_rate'] = (nan_count + below_min + above_max) / len(predictions)
        
        return violations


class QuantilePostProcessor(PostProcessor):
    """Post-processor for quantile predictions ensuring monotonicity."""
    
    def __init__(self, quantiles: list[float], enforce_monotonic: bool = True):
        self.quantiles = sorted(quantiles)
        self.enforce_monotonic = enforce_monotonic
        
    def apply_rules(
        self,
        predictions: Array,
        context: dict[str, Any] | None = None,
    ) -> Array:
        """Apply business rules to quantile predictions."""
        predictions = np.asarray(predictions, dtype=float)
        
        if predictions.ndim != 2 or predictions.shape[1] != len(self.quantiles):
            raise ValueError(f"Expected shape (n_samples, {len(self.quantiles)}), got {predictions.shape}")
            
        # Handle NaN/inf values
        predictions = np.nan_to_num(predictions, nan=0.0)
        
        # Enforce monotonicity if requested
        if self.enforce_monotonic:
            predictions = np.sort(predictions, axis=1)
            
        return predictions
        
    def validate_predictions(self, predictions: Array) -> bool:
        """Return True if predictions satisfy business constraints."""
        predictions = np.asarray(predictions)
        
        # Check for NaN/inf values
        if not np.all(np.isfinite(predictions)):
            return False
            
        # Check monotonicity
        if self.enforce_monotonic:
            for row in predictions:
                if not np.all(row[:-1] <= row[1:]):
                    return False
                    
        return True
        
    def report_violations(self, predictions: Array) -> dict[str, Any]:
        """Return a structured report of constraint violations."""
        predictions = np.asarray(predictions)
        violations = {}
        
        # Count NaN/inf values
        nan_count = np.sum(~np.isfinite(predictions))
        if nan_count > 0:
            violations['nan_inf_count'] = int(nan_count)
            
        # Count monotonicity violations
        monotonicity_violations = 0
        for row in predictions:
            if not np.all(row[:-1] <= row[1:]):
                monotonicity_violations += 1
                
        if monotonicity_violations > 0:
            violations['monotonicity_violations'] = monotonicity_violations
            
        violations['total_samples'] = len(predictions)
        violations['violation_rate'] = (nan_count + monotonicity_violations) / len(predictions)
        
        return violations

"""
Task configurations for different modeling tasks.

This module provides task-specific configurations that define defaults
and factory methods for different modeling scenarios.
"""

from stock_market_analytics.config import config
from stock_market_analytics.modeling.pipeline_components.protocols import (
    TaskConfigProtocol,
    TaskType,
)
from stock_market_analytics.modeling.pipeline_components.evaluation.evaluators import QuantileRegressionEvaluator
from stock_market_analytics.modeling.pipeline_components.calibration.calibrators import QuantileIntervalCalibrator


class QuantileRegressionTaskConfig(TaskConfigProtocol):
    """
    Task configuration for quantile regression modeling.
    
    Provides defaults and factory methods for quantile regression tasks.
    """
    
    @property
    def task_type(self) -> TaskType:
        """Return the task type."""
        return "quantile_regression"
    
    @property
    def primary_metric_name(self) -> str:
        """Return the primary metric for optimization."""
        return "pinball_mean"
    
    @property
    def default_quantiles(self) -> list[float]:
        """Return default quantiles for this task."""
        return config.modeling.quantiles
    
    @property
    def prefers_calibration(self) -> bool:
        """Whether this task benefits from calibration."""
        return True
    
    def make_default_evaluator(self) -> QuantileRegressionEvaluator:
        """Create a default evaluator for this task."""
        return QuantileRegressionEvaluator(
            quantiles=self.default_quantiles,
            target_coverage=config.modeling.target_coverage,
        )
    
    def make_default_calibrator(self) -> QuantileIntervalCalibrator:
        """Create a default calibrator for this task."""
        return QuantileIntervalCalibrator(
            target_coverage=config.modeling.target_coverage,
            low_quantile=self.default_quantiles[0],    # First quantile
            high_quantile=self.default_quantiles[-1],  # Last quantile
        )
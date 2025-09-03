"""
Evaluation configuration classes.

This module defines configuration classes for model evaluation,
conformal prediction, and baseline comparison.
"""

from typing import List, Tuple
from pydantic import BaseModel, Field


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""
    
    # Conformal prediction
    target_coverage: float = Field(
        default=0.8,
        description="Target coverage for conformal prediction intervals"
    )
    
    coverage_interval: Tuple[float, float] = Field(
        default=(0.1, 0.9),
        description="Quantile interval for coverage calculation"
    )
    
    # Quantile indices (computed from quantiles)
    low_idx: int = Field(
        default=0,
        description="Index of low quantile for interval"
    )
    mid_idx: int = Field(
        default=2, 
        description="Index of median quantile"
    )
    high_idx: int = Field(
        default=4,
        description="Index of high quantile for interval"
    )
    
    # Baseline comparison
    baseline_strategies: List[str] = Field(
        default=[
            "historical_quantiles",
            "random_walk",
            "random_noise", 
            "seasonal_naive"
        ],
        description="Baseline strategies to evaluate"
    )
    
    # Evaluation metrics
    return_per_quantile_metrics: bool = Field(
        default=True,
        description="Whether to compute per-quantile metrics"
    )
    
    lambda_crossing_penalty: float = Field(
        default=0.0,
        description="Penalty for quantile crossing violations"
    )
    
    def update_quantile_indices(self, quantiles: List[float]) -> None:
        """Update quantile indices based on provided quantiles."""
        n_quantiles = len(quantiles)
        self.low_idx = 0
        self.high_idx = n_quantiles - 1
        self.mid_idx = n_quantiles // 2
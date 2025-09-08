"""
Protocol-compliant evaluators for stock market analytics models.

This module provides evaluators that work with PredictionBundle objects,
enabling model-agnostic evaluation and easy experimentation.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from stock_market_analytics.config import config
from stock_market_analytics.modeling.pipeline_components.evaluation.evaluation_functions import (
    coverage,
    eval_multiquantile,
    mean_width,
)
from stock_market_analytics.modeling.pipeline_components.protocols import (
    EvaluationResult,
    ModelEvaluatorProtocol,
    PredictionBundle,
    QuantilePreds,
    IntervalPreds,
    is_quantiles,
    is_interval,
    TaskType,
    PredKind,
)

NDArrayF = npt.NDArray[np.float64]


class QuantileRegressionEvaluator(ModelEvaluatorProtocol):
    """
    Protocol-compliant evaluator for quantile regression models.
    
    This evaluator consumes PredictionBundle objects instead of raw model outputs,
    making it model-agnostic and enabling easy experimentation with different models.
    """
    
    task_type: TaskType = "quantile_regression"
    accepted_kinds: tuple[PredKind, ...] = ("quantiles", "interval")
    
    def __init__(
        self,
        quantiles: list[float] | None = None,
        interval: tuple[float, float] | None = None,
        target_coverage: float | None = None,
    ):
        self.quantiles = quantiles or config.modeling.quantiles
        self.interval = interval or (0.1, 0.9)  # Default 80% interval
        self.target_coverage = target_coverage or config.modeling.target_coverage
    
    def evaluate_predictions(
        self,
        y_true: NDArrayF,
        preds: PredictionBundle,
        *,
        sample_weight: NDArrayF | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluate predictions using the appropriate metrics for the prediction type.
        
        Args:
            y_true: Ground truth targets
            preds: Prediction bundle (quantiles or intervals)
            sample_weight: Optional sample weights
            **kwargs: Additional evaluation parameters (lambda_cross, etc.)
        
        Returns:
            EvaluationResult with task-specific metrics
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        
        if is_quantiles(preds):
            return self._evaluate_quantile_predictions(y_true, preds, sample_weight, **kwargs)
        elif is_interval(preds):
            return self._evaluate_interval_predictions(y_true, preds, sample_weight, **kwargs)
        else:
            raise ValueError(
                f"Evaluator accepts {self.accepted_kinds} predictions, "
                f"but received {preds.kind}"
            )
    
    def _evaluate_quantile_predictions(
        self,
        y_true: NDArrayF,
        preds: QuantilePreds,
        sample_weight: NDArrayF | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate multi-quantile predictions using pinball loss and coverage."""
        lambda_cross = kwargs.get("lambda_cross", 0.0)
        return_per_quantile = kwargs.get("return_per_quantile", False)
        
        # Use existing eval_multiquantile function
        loss, metrics = eval_multiquantile(
            y_true=y_true,
            q_pred=preds.q,
            quantiles=list(preds.quantiles),
            interval=self.interval,
            sample_weight=sample_weight,
            lambda_cross=lambda_cross,
            return_per_quantile=return_per_quantile,
        )
        
        return EvaluationResult(
            task_type=self.task_type,
            primary_metric_name="pinball_mean",
            primary_metric_value=metrics["pinball_mean"],
            metrics=metrics,
            n_samples=len(y_true),
            artifacts={},
        )
    
    def _evaluate_interval_predictions(
        self,
        y_true: NDArrayF,
        preds: IntervalPreds,
        sample_weight: NDArrayF | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate interval predictions using coverage and width metrics."""
        # Calculate coverage and width
        cov = coverage(y_true, preds.lo, preds.hi)
        width = mean_width(preds.lo, preds.hi)
        
        # Calculate coverage gap (how far from target coverage)
        target_cov = preds.target_coverage or self.target_coverage
        coverage_gap = abs(cov - target_cov)
        
        metrics = {
            "coverage": cov,
            "mean_width": width,
            "target_coverage": target_cov,
            "coverage_gap": coverage_gap,
        }
        
        return EvaluationResult(
            task_type=self.task_type,
            primary_metric_name="coverage_gap",
            primary_metric_value=coverage_gap,
            metrics=metrics,
            n_samples=len(y_true),
            artifacts={},
        )


class EvaluationReport:
    """
    Helper class to generate formatted evaluation reports.
    """

    @staticmethod
    def format_metrics(
        metrics: dict[str, Any], title: str = "Evaluation Metrics"
    ) -> str:
        """Format metrics dictionary into a readable string."""
        lines = [f"=== {title}", "=" * (len(title) + 8)]

        for key, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"\n{key.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        lines.append(f"  {sub_key}: {sub_value:.4f}")
                    else:
                        lines.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, float):
                lines.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")

        return "\n".join(lines)

    @staticmethod
    def print_summary(evaluation_results: dict[str, Any]) -> None:
        """Print a formatted summary of evaluation results."""
        if "training" in evaluation_results:
            print(
                EvaluationReport.format_metrics(
                    evaluation_results["training"]["metrics"], "Training Metrics"
                )
            )

        if "calibrated" in evaluation_results:
            print(
                "\n"
                + EvaluationReport.format_metrics(
                    evaluation_results["calibrated"], "Calibrated Evaluation"
                )
            )
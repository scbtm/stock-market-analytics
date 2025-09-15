"""
Model evaluator implementations for standardized performance assessment.

This module contains evaluator classes that implement the evaluation protocols
and provide standardized metrics for different modeling tasks.
"""

from collections.abc import Sequence

import numpy as np

from stock_market_analytics.modeling.model_factory.evaluation.evaluation_functions import (
    align_predictions_to_quantiles,
    crps_from_quantiles,
    drop_nan_rows_for_quantiles,
    ensure_sorted_unique_quantiles,
    interval_score,
    mean_interval_width,
    monotonicity_violation_rate,
    normalized_interval_width,
    pinball_loss_vectorized,
    pit_ece,
    pit_ks_statistic,
    pit_values,
    prediction_interval_coverage_probability,
    quantile_coverage,
    quantile_loss_differential,
    validate_xyq_shapes,
)
from stock_market_analytics.modeling.model_factory.protocols import (
    Array,
    QuantileEvaluator,
)

# -------------------------
# Evaluator
# -------------------------


class QuantileRegressionEvaluator(QuantileEvaluator):
    """
    Evaluator for quantile regression tasks.

    Provides metrics specific to quantile predictions including
    pinball loss, coverage, CRPS, PIT calibration, and crossing diagnostics.
    """

    def __init__(self, quantiles: Sequence[float]):
        """
        Initialize the quantile regression evaluator.

        Args:
            quantiles: List of quantile levels being predicted
        """
        self.quantiles = ensure_sorted_unique_quantiles(
            np.array(quantiles, dtype=float)
        )

    def evaluate_quantiles(
        self, y_true: Array, y_pred_quantiles: Array, quantiles: Sequence[float]
    ) -> dict[str, float]:
        """
        Evaluate quantile predictions.
        """
        validate_xyq_shapes(y_true, np.asarray(y_pred_quantiles), np.asarray(quantiles))
        # Align and drop any NaN rows to avoid biased metrics
        yq, q_sorted, _ = align_predictions_to_quantiles(
            y_pred_quantiles, np.array(quantiles, dtype=float)
        )
        y_true = np.asarray(y_true).reshape(-1)
        y_true, yq, n_dropped = drop_nan_rows_for_quantiles(y_true, yq)
        n = y_true.shape[0]

        # Basic metrics
        pinball = pinball_loss_vectorized(y_true, yq, q_sorted)
        cov = quantile_coverage(y_true, yq, q_sorted)
        qld = quantile_loss_differential(y_true, yq, q_sorted)
        crps = crps_from_quantiles(y_true, yq, q_sorted)

        # Crossing diagnostics
        cross_rate, cross_count = monotonicity_violation_rate(yq)

        # PIT-based distributional calibration
        pit = pit_values(y_true, yq, q_sorted)
        pit_mean = float(np.mean(pit)) if pit.size else float("nan")
        pit_std = float(np.std(pit)) if pit.size else float("nan")
        pit_ks = pit_ks_statistic(pit)
        pit_ece20 = pit_ece(pit, n_bins=20)

        metrics: dict[str, float] = {
            "n_samples_evaluated": float(n),
            "n_rows_dropped_nan": float(n_dropped),
            "mean_pinball_loss": float(np.mean(pinball)),
            "median_pinball_loss": float(np.median(pinball)),
            "max_pinball_loss": float(np.max(pinball)),
            "mean_coverage": float(np.mean(cov)),
            "coverage_deviation": float(np.mean(np.abs(cov - q_sorted))),
            "crps": crps,
            "monotonicity_violation_rate": float(cross_rate),
            "monotonicity_violated_rows": float(cross_count),
            # PIT calibration summaries (Uniform(0,1) target)
            "pit_mean": pit_mean,  # ~0.5 if calibrated
            "pit_std": pit_std,  # ~1/sqrt(12) ≈ 0.2887
            "pit_ks": pit_ks,  # lower is better
            "pit_ece_20bins": pit_ece20,  # lower is better
        }

        # Per-quantile metrics
        for i, q in enumerate(q_sorted):
            pct = int(round(q * 100))
            metrics[f"pinball_loss_q{pct}"] = float(pinball[i])
            metrics[f"coverage_q{pct}"] = float(cov[i])
            metrics[f"coverage_error_q{pct}"] = float(abs(cov[i] - q))

        # Differential bundle
        metrics.update(qld)
        return metrics

    def evaluate_intervals(
        self,
        y_true: Array,
        y_lower: Array,
        y_upper: Array,
        alpha: float = 0.2,
    ) -> dict[str, float]:
        """
        Evaluate prediction intervals.
        """
        return {
            "interval_score": interval_score(y_true, y_lower, y_upper, alpha),
            "coverage_probability": prediction_interval_coverage_probability(
                y_true, y_lower, y_upper
            ),
            "mean_interval_width": mean_interval_width(y_lower, y_upper),
            "normalized_interval_width": normalized_interval_width(
                y_lower, y_upper, y_true
            ),
        }

    def get_metric_names(self) -> list[str]:
        names = [
            "n_samples_evaluated",
            "n_rows_dropped_nan",
            "mean_pinball_loss",
            "median_pinball_loss",
            "max_pinball_loss",
            "mean_coverage",
            "coverage_deviation",
            "mean_coverage_error",
            "max_coverage_error",
            "coverage_bias",
            "crps",
            "monotonicity_violation_rate",
            "monotonicity_violated_rows",
            "pit_mean",
            "pit_std",
            "pit_ks",
            "pit_ece_20bins",
        ]
        for q in self.quantiles:
            pct = int(round(float(q) * 100))
            names += [
                f"pinball_loss_q{pct}",
                f"coverage_q{pct}",
                f"coverage_error_q{pct}",
            ]
        return names

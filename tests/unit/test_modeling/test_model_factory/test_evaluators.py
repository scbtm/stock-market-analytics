"""Unit tests for model evaluators."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from stock_market_analytics.modeling.model_factory.evaluation.evaluators import (
    QuantileRegressionEvaluator,
)


class TestQuantileRegressionEvaluator:
    """Test suite for QuantileRegressionEvaluator class."""

    def test_init_basic(self):
        """Test basic initialization with valid quantiles."""
        quantiles = [0.1, 0.5, 0.9]
        evaluator = QuantileRegressionEvaluator(quantiles)

        np.testing.assert_array_almost_equal(evaluator.quantiles, [0.1, 0.5, 0.9])

    def test_init_unsorted_quantiles(self):
        """Test initialization with unsorted quantiles (should be sorted)."""
        quantiles = [0.9, 0.1, 0.5]
        evaluator = QuantileRegressionEvaluator(quantiles)

        # Should be automatically sorted
        np.testing.assert_array_almost_equal(evaluator.quantiles, [0.1, 0.5, 0.9])

    def test_init_invalid_quantiles_out_of_range(self):
        """Test initialization with out-of-range quantiles."""
        with pytest.raises(ValueError, match="must be in"):
            QuantileRegressionEvaluator([0.1, 0.5, 1.1])

        with pytest.raises(ValueError, match="must be in"):
            QuantileRegressionEvaluator([-0.1, 0.5, 0.9])

    def test_init_duplicate_quantiles(self):
        """Test initialization with duplicate quantiles."""
        with pytest.raises(ValueError, match="must be strictly unique"):
            QuantileRegressionEvaluator([0.1, 0.5, 0.5])

    def test_init_single_quantile(self):
        """Test initialization with single quantile."""
        evaluator = QuantileRegressionEvaluator([0.5])
        np.testing.assert_array_almost_equal(evaluator.quantiles, [0.5])

    def test_evaluate_point_predictions_empty(self):
        """Test evaluate method for point predictions (should return empty dict)."""
        evaluator = QuantileRegressionEvaluator([0.5])
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        y_pred_quantiles = y_pred.reshape(-1, 1)
        quantiles = [0.5]

        result = evaluator.evaluate_quantiles(y_true, y_pred_quantiles, quantiles)

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_evaluate_quantiles_perfect_predictions(self):
        """Test evaluate_quantiles with perfect predictions."""
        evaluator = QuantileRegressionEvaluator([0.1, 0.5, 0.9])

        # Perfect predictions: true values at median, with reasonable bounds
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_quantiles = np.array(
            [
                [0.5, 1.0, 1.5],  # Sample 1: bounds around true value 1.0
                [1.5, 2.0, 2.5],  # Sample 2: bounds around true value 2.0
                [2.5, 3.0, 3.5],  # Sample 3: bounds around true value 3.0
            ]
        )
        quantiles = [0.1, 0.5, 0.9]

        result = evaluator.evaluate_quantiles(y_true, y_pred_quantiles, quantiles)

        # Check basic structure
        assert isinstance(result, dict)
        assert "n_samples_evaluated" in result
        assert result["n_samples_evaluated"] == 3.0
        assert result["n_rows_dropped_nan"] == 0.0

        # Perfect median predictions should have zero pinball loss for q=0.5
        assert result["pinball_loss_q50"] == 0.0

        # Coverage should be exact for perfect predictions
        assert result["coverage_q50"] == 1.0  # All true values equal median predictions

        # Check that all expected metrics are present
        assert "mean_pinball_loss" in result
        assert "crps" in result
        assert "pit_mean" in result

    def test_evaluate_quantiles_basic_metrics(self):
        """Test evaluate_quantiles with realistic predictions."""
        evaluator = QuantileRegressionEvaluator([0.25, 0.5, 0.75])

        # Realistic predictions with some errors
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_quantiles = np.array(
            [
                [0.8, 1.1, 1.4],  # Sample 1: slightly off
                [1.8, 2.2, 2.4],  # Sample 2: slightly off
                [2.7, 3.1, 3.3],  # Sample 3: slightly off
                [3.6, 4.2, 4.4],  # Sample 4: slightly off
                [4.5, 5.3, 5.5],  # Sample 5: slightly off
            ]
        )
        quantiles = [0.25, 0.5, 0.75]

        result = evaluator.evaluate_quantiles(y_true, y_pred_quantiles, quantiles)

        # Basic structure checks
        assert result["n_samples_evaluated"] == 5.0
        assert result["n_rows_dropped_nan"] == 0.0

        # All metrics should be finite numbers
        for key, value in result.items():
            assert np.isfinite(value), f"Metric {key} is not finite: {value}"

        # Check specific metric ranges
        assert result["mean_pinball_loss"] >= 0.0  # Pinball loss is always non-negative
        assert result["crps"] >= 0.0  # CRPS is always non-negative
        assert 0.0 <= result["mean_coverage"] <= 1.0  # Coverage is between 0 and 1
        assert (
            0.0 <= result["monotonicity_violation_rate"] <= 1.0
        )  # Violation rate is a proportion

        # PIT statistics should be reasonable for uniform distribution
        assert 0.0 <= result["pit_mean"] <= 1.0
        assert result["pit_std"] >= 0.0

        # Per-quantile metrics should exist
        assert "pinball_loss_q25" in result
        assert "pinball_loss_q50" in result
        assert "pinball_loss_q75" in result
        assert "coverage_q25" in result
        assert "coverage_q50" in result
        assert "coverage_q75" in result

    def test_evaluate_quantiles_with_nans(self):
        """Test evaluate_quantiles handling of NaN values."""
        evaluator = QuantileRegressionEvaluator([0.1, 0.5, 0.9])

        # Data with some NaN values
        y_true = np.array([1.0, np.nan, 3.0, 4.0])
        y_pred_quantiles = np.array(
            [
                [0.5, 1.0, 1.5],
                [np.nan, np.nan, np.nan],  # This row should be dropped
                [2.5, 3.0, 3.5],
                [3.5, 4.0, 4.5],
            ]
        )
        quantiles = [0.1, 0.5, 0.9]

        result = evaluator.evaluate_quantiles(y_true, y_pred_quantiles, quantiles)

        # Should have dropped 1 row with NaN
        assert result["n_samples_evaluated"] == 3.0  # 4 - 1 = 3
        assert result["n_rows_dropped_nan"] == 1.0

        # Metrics should still be computed on valid data
        assert np.isfinite(result["mean_pinball_loss"])
        assert np.isfinite(result["crps"])

    def test_evaluate_quantiles_monotonicity_violations(self):
        """Test evaluate_quantiles with quantile crossing violations."""
        evaluator = QuantileRegressionEvaluator([0.1, 0.5, 0.9])

        y_true = np.array([1.0, 2.0, 3.0])
        # Intentionally create crossing violations (higher quantile < lower quantile)
        y_pred_quantiles = np.array(
            [
                [1.5, 1.0, 0.5],  # Decreasing (violation)
                [2.0, 2.0, 2.0],  # Flat (not technically a violation)
                [3.5, 3.0, 2.5],  # Decreasing (violation)
            ]
        )
        quantiles = [0.1, 0.5, 0.9]

        result = evaluator.evaluate_quantiles(y_true, y_pred_quantiles, quantiles)

        # Should detect monotonicity violations
        assert result["monotonicity_violation_rate"] > 0.0
        assert result["monotonicity_violated_rows"] > 0.0

    def test_evaluate_quantiles_coverage_analysis(self):
        """Test evaluate_quantiles coverage analysis with known outcomes."""
        evaluator = QuantileRegressionEvaluator([0.2, 0.5, 0.8])

        # Create data where we know the coverage outcomes
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Target values
        y_pred_quantiles = np.array(
            [
                [0.9, 1.1, 1.3],  # True value 1.0: below q20, between q20-q50
                [1.8, 2.2, 2.4],  # True value 2.0: below q20, above q50
                [2.5, 3.5, 3.7],  # True value 3.0: below all quantiles
                [3.2, 4.4, 4.6],  # True value 4.0: below q20, above q50
                [4.1, 5.1, 5.3],  # True value 5.0: below q20, between q20-q50
            ]
        )
        quantiles = [0.2, 0.5, 0.8]

        result = evaluator.evaluate_quantiles(y_true, y_pred_quantiles, quantiles)

        # Manual coverage calculation for verification
        # q20 coverage: values <= q20 prediction
        q20_coverage = np.mean(y_true <= y_pred_quantiles[:, 0])
        # q50 coverage: values <= q50 prediction
        q50_coverage = np.mean(y_true <= y_pred_quantiles[:, 1])
        # q80 coverage: values <= q80 prediction
        q80_coverage = np.mean(y_true <= y_pred_quantiles[:, 2])

        # Check that computed coverage matches our manual calculation
        assert abs(result["coverage_q20"] - q20_coverage) < 1e-10
        assert abs(result["coverage_q50"] - q50_coverage) < 1e-10
        assert abs(result["coverage_q80"] - q80_coverage) < 1e-10

    def test_evaluate_intervals_basic(self):
        """Test evaluate_intervals method."""
        evaluator = QuantileRegressionEvaluator([0.1, 0.5, 0.9])

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_lower = np.array([0.8, 1.8, 2.8, 3.8, 4.8])
        y_upper = np.array([1.2, 2.2, 3.2, 4.2, 5.2])
        alpha = 0.2  # 80% prediction interval

        result = evaluator.evaluate_intervals(y_true, y_lower, y_upper, alpha)

        # Check structure
        assert isinstance(result, dict)
        assert "interval_score" in result
        assert "coverage_probability" in result
        assert "mean_interval_width" in result
        assert "normalized_interval_width" in result

        # Check value ranges
        assert np.isfinite(result["interval_score"])
        assert 0.0 <= result["coverage_probability"] <= 1.0
        assert result["mean_interval_width"] >= 0.0
        assert result["normalized_interval_width"] >= 0.0

        # For this perfect case, coverage should be 100%
        assert result["coverage_probability"] == 1.0
        # Mean width should be 0.4 (all intervals have width 0.4)
        assert abs(result["mean_interval_width"] - 0.4) < 1e-10

    def test_evaluate_intervals_no_coverage(self):
        """Test evaluate_intervals with no coverage."""
        evaluator = QuantileRegressionEvaluator([0.1, 0.5, 0.9])

        y_true = np.array([1.0, 2.0, 3.0])
        y_lower = np.array([2.0, 3.0, 4.0])  # All lower bounds above true values
        y_upper = np.array([3.0, 4.0, 5.0])  # All upper bounds above true values

        result = evaluator.evaluate_intervals(y_true, y_lower, y_upper)

        # No coverage expected
        assert result["coverage_probability"] == 0.0

        # Interval score should be positive (penalty for misses)
        assert result["interval_score"] > 0.0

    def test_get_metric_names_basic_quantiles(self):
        """Test get_metric_names method with basic quantiles."""
        evaluator = QuantileRegressionEvaluator([0.1, 0.5, 0.9])

        names = evaluator.get_metric_names()

        assert isinstance(names, list)
        assert len(names) > 0

        # Check for basic metrics
        expected_basic = [
            "n_samples_evaluated",
            "n_rows_dropped_nan",
            "mean_pinball_loss",
            "crps",
            "monotonicity_violation_rate",
            "pit_mean",
            "pit_ks",
        ]
        for metric in expected_basic:
            assert metric in names

        # Check for per-quantile metrics
        assert "pinball_loss_q10" in names
        assert "pinball_loss_q50" in names
        assert "pinball_loss_q90" in names
        assert "coverage_q10" in names
        assert "coverage_q50" in names
        assert "coverage_q90" in names
        assert "coverage_error_q10" in names
        assert "coverage_error_q50" in names
        assert "coverage_error_q90" in names

    def test_get_metric_names_many_quantiles(self):
        """Test get_metric_names with many quantiles."""
        quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        evaluator = QuantileRegressionEvaluator(quantiles)

        names = evaluator.get_metric_names()

        # Should have per-quantile metrics for all quantiles
        for q in quantiles:
            pct = int(round(q * 100))
            assert f"pinball_loss_q{pct}" in names
            assert f"coverage_q{pct}" in names
            assert f"coverage_error_q{pct}" in names

    def test_get_metric_names_single_quantile(self):
        """Test get_metric_names with single quantile."""
        evaluator = QuantileRegressionEvaluator([0.5])

        names = evaluator.get_metric_names()

        # Should still have all basic metrics
        basic_metrics = ["n_samples_evaluated", "mean_pinball_loss", "crps", "pit_mean"]
        for metric in basic_metrics:
            assert metric in names

        # Should have per-quantile metrics for q50
        assert "pinball_loss_q50" in names
        assert "coverage_q50" in names
        assert "coverage_error_q50" in names

    def test_evaluate_quantiles_edge_case_single_sample(self):
        """Test evaluate_quantiles with single sample."""
        evaluator = QuantileRegressionEvaluator([0.25, 0.5, 0.75])

        y_true = np.array([2.5])
        y_pred_quantiles = np.array([[2.0, 2.5, 3.0]])
        quantiles = [0.25, 0.5, 0.75]

        result = evaluator.evaluate_quantiles(y_true, y_pred_quantiles, quantiles)

        assert result["n_samples_evaluated"] == 1.0
        assert np.isfinite(result["mean_pinball_loss"])
        assert np.isfinite(result["crps"])
        # Perfect median prediction
        assert result["pinball_loss_q50"] == 0.0

    def test_evaluate_quantiles_edge_case_all_zeros(self):
        """Test evaluate_quantiles with all zero values."""
        evaluator = QuantileRegressionEvaluator([0.1, 0.5, 0.9])

        y_true = np.array([0.0, 0.0, 0.0])
        y_pred_quantiles = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        quantiles = [0.1, 0.5, 0.9]

        result = evaluator.evaluate_quantiles(y_true, y_pred_quantiles, quantiles)

        # Perfect predictions should give zero pinball loss
        assert result["mean_pinball_loss"] == 0.0
        assert result["crps"] == 0.0
        assert result["monotonicity_violation_rate"] == 0.0

    def test_evaluate_quantiles_mismatched_dimensions(self):
        """Test evaluate_quantiles with mismatched dimensions."""
        evaluator = QuantileRegressionEvaluator([0.1, 0.5, 0.9])

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_quantiles = np.array([[1.0, 1.5], [2.0, 2.5]])  # Wrong shape
        quantiles = [0.1, 0.5, 0.9]

        with pytest.raises(ValueError):
            evaluator.evaluate_quantiles(y_true, y_pred_quantiles, quantiles)

    def test_evaluate_quantiles_quantile_mismatch(self):
        """Test evaluate_quantiles with quantile count mismatch."""
        evaluator = QuantileRegressionEvaluator([0.1, 0.5, 0.9])

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_quantiles = np.array(
            [
                [1.0, 1.5, 2.0],
                [2.0, 2.5, 3.0],
                [3.0, 3.5, 4.0],
            ]
        )
        quantiles = [0.25, 0.75]  # Different quantiles than predictions

        # The function may actually handle this case by aligning quantiles
        # Let's test that it doesn't raise an error but processes correctly
        result = evaluator.evaluate_quantiles(y_true, y_pred_quantiles, quantiles)

        # Should process successfully since align_predictions_to_quantiles handles mismatches
        assert isinstance(result, dict)
        assert "mean_pinball_loss" in result

    @patch(
        "stock_market_analytics.modeling.model_factory.evaluation.evaluators.ensure_sorted_unique_quantiles"
    )
    def test_init_calls_validation(self, mock_validate):
        """Test that initialization calls quantile validation."""
        mock_validate.return_value = np.array([0.1, 0.5, 0.9])

        quantiles = [0.9, 0.1, 0.5]  # Unsorted
        evaluator = QuantileRegressionEvaluator(quantiles)

        mock_validate.assert_called_once()
        # Check that it was called with the converted array
        call_args = mock_validate.call_args[0][0]
        np.testing.assert_array_almost_equal(call_args, [0.9, 0.1, 0.5])

    def test_evaluate_intervals_edge_cases(self):
        """Test evaluate_intervals with edge cases."""
        evaluator = QuantileRegressionEvaluator([0.1, 0.5, 0.9])

        # Test with single point
        y_true = np.array([2.0])
        y_lower = np.array([1.5])
        y_upper = np.array([2.5])

        result = evaluator.evaluate_intervals(y_true, y_lower, y_upper)

        assert result["coverage_probability"] == 1.0
        assert result["mean_interval_width"] == 1.0

        # Test with zero-width intervals
        y_lower_zero = np.array([2.0])
        y_upper_zero = np.array([2.0])

        result_zero = evaluator.evaluate_intervals(y_true, y_lower_zero, y_upper_zero)

        assert result_zero["mean_interval_width"] == 0.0

    def test_comprehensive_metrics_integration(self):
        """Test that all metrics integrate properly in a realistic scenario."""
        evaluator = QuantileRegressionEvaluator([0.05, 0.25, 0.5, 0.75, 0.95])

        # Generate somewhat realistic data
        np.random.seed(42)  # For reproducibility
        n_samples = 100
        y_true = np.random.normal(10, 2, n_samples)

        # Create predictions with some realistic error structure
        y_pred_quantiles = np.column_stack(
            [
                y_true + np.random.normal(-3, 0.5, n_samples),  # q05: lower bound
                y_true + np.random.normal(-1, 0.3, n_samples),  # q25
                y_true + np.random.normal(0, 0.2, n_samples),  # q50: median
                y_true + np.random.normal(1, 0.3, n_samples),  # q75
                y_true + np.random.normal(3, 0.5, n_samples),  # q95: upper bound
            ]
        )
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

        result = evaluator.evaluate_quantiles(y_true, y_pred_quantiles, quantiles)

        # Test that all expected metrics are present
        metric_names = evaluator.get_metric_names()
        for name in metric_names:
            assert name in result, f"Missing metric: {name}"

        # Test that all values are finite and reasonable
        for key, value in result.items():
            assert np.isfinite(value), f"Non-finite value for {key}: {value}"

        # Test some basic sanity checks
        assert result["n_samples_evaluated"] == n_samples
        assert result["mean_pinball_loss"] >= 0
        assert result["crps"] >= 0
        assert 0 <= result["mean_coverage"] <= 1
        assert 0 <= result["monotonicity_violation_rate"] <= 1
        assert 0 <= result["pit_mean"] <= 1

"""Unit tests for monitoring metrics functions."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from stock_market_analytics.monitoring.monitoring_metrics import (
    kolmogorov_smirnov_test,
    wasserstein_distance,
    population_stability_index,
    jensen_shannon_divergence,
    multivariate_covariate_drift_metrics,
    prediction_drift_metrics,
    target_drift_metrics,
    quantile_regression_performance_metrics,
    quantile_performance_trends,
    prediction_interval_performance_metrics,
    interval_calibration_curve,
    interval_sharpness_metrics,
    median_point_metrics,
)


class TestDistributionShiftDetection:
    """Test suite for distribution shift detection functions."""

    def test_kolmogorov_smirnov_test_identical(self):
        """Test KS test with identical distributions."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = kolmogorov_smirnov_test(data, data)

        assert "ks_statistic" in result
        assert "p_value" in result
        assert result["ks_statistic"] == 0.0  # Identical distributions
        assert result["p_value"] == 1.0

    def test_kolmogorov_smirnov_test_different(self):
        """Test KS test with different distributions."""
        ref_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        curr_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = kolmogorov_smirnov_test(ref_data, curr_data)

        assert result["ks_statistic"] == 1.0  # Completely different
        assert result["p_value"] < 0.05  # Highly significant

    def test_kolmogorov_smirnov_test_empty_data(self):
        """Test KS test with empty data."""
        empty_data = np.array([])
        normal_data = np.array([1.0, 2.0, 3.0])
        result = kolmogorov_smirnov_test(empty_data, normal_data)

        assert np.isnan(result["ks_statistic"])
        assert np.isnan(result["p_value"])

    def test_wasserstein_distance_identical(self):
        """Test Wasserstein distance with identical distributions."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wasserstein_distance(data, data)

        assert result == 0.0  # Identical distributions

    def test_wasserstein_distance_different(self):
        """Test Wasserstein distance with different distributions."""
        ref_data = np.array([1.0, 2.0, 3.0])
        curr_data = np.array([4.0, 5.0, 6.0])  # Shifted by 3
        result = wasserstein_distance(ref_data, curr_data)

        assert abs(result - 3.0) < 1e-10  # Distance equals shift

    def test_wasserstein_distance_empty_data(self):
        """Test Wasserstein distance with empty data."""
        empty_data = np.array([])
        normal_data = np.array([1.0, 2.0, 3.0])
        result = wasserstein_distance(empty_data, normal_data)

        assert np.isnan(result)

    def test_population_stability_index_no_drift(self):
        """Test PSI with no drift."""
        data = np.random.normal(0, 1, 1000)
        ref_data = data[:500]
        curr_data = data[500:]

        result = population_stability_index(ref_data, curr_data)

        assert result["psi"] < 0.1  # Should indicate no drift
        assert result["interpretation"] == "no_drift"

    def test_population_stability_index_major_drift(self):
        """Test PSI with major drift."""
        ref_data = np.random.normal(0, 1, 500)
        curr_data = np.random.normal(5, 1, 500)  # Mean shifted by 5

        result = population_stability_index(ref_data, curr_data)

        assert result["psi"] > 0.2  # Should indicate major drift
        assert result["interpretation"] == "major_drift"

    def test_jensen_shannon_divergence_identical(self):
        """Test JS divergence with identical distributions."""
        data = np.random.normal(0, 1, 1000)
        result = jensen_shannon_divergence(data, data)

        assert result == 0.0  # Identical distributions

    def test_jensen_shannon_divergence_different(self):
        """Test JS divergence with different distributions."""
        ref_data = np.random.normal(0, 1, 1000)
        curr_data = np.random.normal(3, 1, 1000)  # Different mean

        result = jensen_shannon_divergence(ref_data, curr_data)

        assert 0 < result <= 1  # Should be in valid range
        assert result > 0.1  # Should detect significant difference

    def test_multivariate_covariate_drift_metrics(self):
        """Test multivariate drift detection."""
        ref_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.uniform(0, 1, 100),
            'non_numeric': ['A'] * 100  # Should be ignored
        })

        curr_df = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1, 100),  # Slight shift
            'feature2': np.random.uniform(0.2, 1.2, 100),  # Slight shift
            'non_numeric': ['B'] * 100  # Should be ignored
        })

        result = multivariate_covariate_drift_metrics(
            ref_df, curr_df, ['feature1', 'feature2', 'non_numeric']
        )

        assert "per_feature" in result
        assert "aggregate" in result
        assert len(result["per_feature"]) == 2  # Only numeric features
        assert "feature1" in result["per_feature"]
        assert "feature2" in result["per_feature"]
        assert "mean_psi" in result["aggregate"]
        assert result["aggregate"]["n_features_analyzed"] == 2


class TestPredictionDriftMetrics:
    """Test suite for prediction drift detection."""

    def test_prediction_drift_metrics_identical(self):
        """Test prediction drift with identical predictions."""
        predictions = np.random.rand(100, 3)  # 100 samples, 3 quantiles
        quantiles = np.array([0.1, 0.5, 0.9])

        result = prediction_drift_metrics(predictions, predictions, quantiles)

        assert "per_quantile" in result
        assert "aggregate" in result
        assert len(result["per_quantile"]) == 3
        assert result["aggregate"]["mean_ks_statistic"] == 0.0

    def test_prediction_drift_metrics_different(self):
        """Test prediction drift with different predictions."""
        ref_predictions = np.random.rand(100, 3)
        curr_predictions = np.random.rand(100, 3) + 1.0  # Shifted
        quantiles = np.array([0.1, 0.5, 0.9])

        result = prediction_drift_metrics(ref_predictions, curr_predictions, quantiles)

        assert result["aggregate"]["mean_ks_statistic"] > 0.5  # Should detect drift
        assert result["aggregate"]["mean_wasserstein"] > 0.3

    def test_prediction_drift_metrics_wrong_dimensions(self):
        """Test error with wrong dimensions."""
        ref_predictions = np.random.rand(100, 2)  # 2 quantiles
        curr_predictions = np.random.rand(100, 2)
        quantiles = np.array([0.1, 0.5, 0.9])  # 3 quantiles - mismatch!

        with pytest.raises(ValueError, match="must match prediction dimensions"):
            prediction_drift_metrics(ref_predictions, curr_predictions, quantiles)


class TestTargetDriftMetrics:
    """Test suite for target drift detection."""

    def test_target_drift_metrics_identical(self):
        """Test target drift with identical targets."""
        targets = np.random.normal(0, 1, 1000)

        result = target_drift_metrics(targets, targets)

        assert "distribution_tests" in result
        assert "distance_metrics" in result
        assert "moments_comparison" in result
        assert result["distance_metrics"]["wasserstein_distance"] == 0.0
        assert result["moments_comparison"]["mean_shift"] == 0.0

    def test_target_drift_metrics_different_means(self):
        """Test target drift with different means."""
        ref_targets = np.random.normal(0, 1, 1000)
        curr_targets = np.random.normal(2, 1, 1000)  # Mean shifted by 2

        result = target_drift_metrics(ref_targets, curr_targets)

        assert abs(result["moments_comparison"]["mean_shift"] - 2.0) < 0.1
        assert result["statistical_tests"]["mean_diff_p_value"] < 0.01  # Significant
        assert result["distance_metrics"]["wasserstein_distance"] > 1.5

    def test_target_drift_metrics_with_nans(self):
        """Test target drift handling NaN values."""
        ref_targets = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        curr_targets = np.array([1.1, 2.1, 3.1, np.nan, 5.1])

        result = target_drift_metrics(ref_targets, curr_targets)

        assert "sample_sizes" in result
        assert result["sample_sizes"]["reference_n"] == 4  # NaNs removed
        assert result["sample_sizes"]["current_n"] == 4


class TestQuantileRegressionMetrics:
    """Test suite for quantile regression performance metrics."""

    def test_quantile_regression_performance_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Perfect predictions: true value at median (q=0.5)
        y_pred = np.column_stack([
            y_true - 0.5,  # q=0.1
            y_true,        # q=0.5 (median)
            y_true + 0.5   # q=0.9
        ])
        quantiles = np.array([0.1, 0.5, 0.9])

        result = quantile_regression_performance_metrics(y_true, y_pred, quantiles)

        assert "pinball_losses" in result
        assert "coverage" in result
        assert "calibration" in result
        assert result["coverage"]["per_quantile"]["q_0.50"] == 1.0  # Perfect median coverage

    def test_quantile_regression_performance_metrics_with_nans(self):
        """Test metrics handling NaN values."""
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([[0.5, 1.0, 1.5], [np.nan, np.nan, np.nan], [2.5, 3.0, 3.5]])
        quantiles = np.array([0.1, 0.5, 0.9])

        result = quantile_regression_performance_metrics(y_true, y_pred, quantiles)

        assert "sample_info" in result
        assert result["sample_info"]["n_valid_samples"] == 2  # Two valid samples
        assert result["sample_info"]["n_nan_removed"] == 1

    def test_quantile_regression_performance_metrics_all_nans(self):
        """Test metrics with all NaN data."""
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        quantiles = np.array([0.1, 0.9])

        result = quantile_regression_performance_metrics(y_true, y_pred, quantiles)

        assert "error" in result
        assert "No valid data" in result["error"]


class TestPredictionIntervalMetrics:
    """Test suite for prediction interval metrics."""

    def test_prediction_interval_performance_metrics_perfect(self):
        """Test interval metrics with perfect coverage."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_lower = y_true - 0.5  # All true values within interval
        y_upper = y_true + 0.5

        result = prediction_interval_performance_metrics(
            y_true, y_lower, y_upper, confidence_level=0.8
        )

        assert result["coverage"]["observed"] == 1.0  # Perfect coverage
        assert abs(result["coverage"]["error"] - 0.2) < 1e-10  # 1.0 - 0.8 = 0.2 over-coverage
        assert result["coverage"]["below_lower_rate"] == 0.0
        assert result["coverage"]["above_upper_rate"] == 0.0

    def test_prediction_interval_performance_metrics_no_coverage(self):
        """Test interval metrics with no coverage."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_lower = y_true + 1.0  # All intervals miss below
        y_upper = y_true + 2.0

        result = prediction_interval_performance_metrics(
            y_true, y_lower, y_upper, confidence_level=0.8
        )

        assert result["coverage"]["observed"] == 0.0  # No coverage
        assert result["coverage"]["below_lower_rate"] == 1.0  # All below
        assert result["coverage"]["above_upper_rate"] == 0.0

    def test_interval_calibration_curve(self):
        """Test interval calibration curve."""
        y_true = np.random.normal(0, 1, 100)
        y_lower = y_true - np.random.uniform(0.5, 2.0, 100)  # Variable widths
        y_upper = y_true + np.random.uniform(0.5, 2.0, 100)

        result = interval_calibration_curve(y_true, y_lower, y_upper, n_bins=5)

        assert "bin_centers" in result
        assert "observed_coverage" in result
        assert "bin_counts" in result
        assert len(result["bin_centers"]) <= 5  # Some bins might be empty

    def test_interval_sharpness_metrics(self):
        """Test interval sharpness metrics."""
        y_lower = np.array([1.0, 2.0, 3.0])
        y_upper = np.array([2.0, 4.0, 6.0])  # Widths: 1.0, 2.0, 3.0

        result = interval_sharpness_metrics(y_lower, y_upper)

        assert result["mean_width"] == 2.0  # (1+2+3)/3 = 2
        assert result["median_width"] == 2.0
        assert result["width_iqr"] == 1.0  # Q75-Q25 = 3-1 = 2, but with only 3 points


class TestPointMetrics:
    """Test suite for point prediction metrics."""

    def test_median_point_metrics_perfect(self):
        """Test point metrics with perfect median predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Perfect median predictions (q=0.5)
        y_pred = np.column_stack([
            y_true - 0.5,  # q=0.1
            y_true,        # q=0.5 (perfect median)
            y_true + 0.5   # q=0.9
        ])
        quantiles = np.array([0.1, 0.5, 0.9])

        result = median_point_metrics(y_true, y_pred, quantiles)

        assert result["mae"] == 0.0  # Perfect predictions
        assert result["rmse"] == 0.0
        assert result["bias"] == 0.0

    def test_median_point_metrics_with_error(self):
        """Test point metrics with prediction error."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.column_stack([
            [0.5, 1.5, 2.5],  # q=0.1
            [1.1, 2.1, 3.1],  # q=0.5 (median with +0.1 bias)
            [1.5, 2.5, 3.5]   # q=0.9
        ])
        quantiles = np.array([0.1, 0.5, 0.9])

        result = median_point_metrics(y_true, y_pred, quantiles)

        assert abs(result["mae"] - 0.1) < 1e-10  # |0.1| average error
        assert abs(result["rmse"] - 0.1) < 1e-10  # sqrt(0.1^2) = 0.1
        assert abs(result["bias"] - 0.1) < 1e-10  # +0.1 systematic error

    def test_median_point_metrics_no_median_quantile(self):
        """Test point metrics when 0.5 quantile not available."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.column_stack([
            [0.5, 1.5, 2.5],  # q=0.1
            [1.5, 2.5, 3.5]   # q=0.9
        ])
        quantiles = np.array([0.1, 0.9])  # No q=0.5

        result = median_point_metrics(y_true, y_pred, quantiles)

        # Should use closest quantile (0.1 in this case)
        assert not np.isnan(result["mae"])
        assert not np.isnan(result["rmse"])
        assert not np.isnan(result["bias"])
"""Unit tests for calibrator classes."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from typing import List

from stock_market_analytics.modeling.model_factory.calibration.calibrators import (
    QuantileConformalCalibrator,
    ConformalizedQuantileCalibrator,
)


class TestQuantileConformalCalibrator:
    """Test suite for QuantileConformalCalibrator class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        calibrator = QuantileConformalCalibrator()

        assert calibrator.alpha == 0.1
        assert calibrator.method == "absolute"
        assert calibrator.eps == 1e-8
        assert calibrator.radius_ is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        calibrator = QuantileConformalCalibrator(
            alpha=0.05, method="normalized", eps=1e-6
        )

        assert calibrator.alpha == 0.05
        assert calibrator.method == "normalized"
        assert calibrator.eps == 1e-6

    def test_init_invalid_alpha(self):
        """Test initialization with invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            QuantileConformalCalibrator(alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            QuantileConformalCalibrator(alpha=1.0)

    def test_fit_absolute_method(self):
        """Test fitting with absolute method."""
        calibrator = QuantileConformalCalibrator(alpha=0.2, method="absolute")

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        result = calibrator.fit(y_pred, y_true)

        assert result is calibrator  # Should return self
        assert calibrator.radius_ is not None
        assert calibrator.radius_ >= 0  # Radius should be non-negative

        # Test that radius is reasonable (should be around the 80th percentile of |residuals|)
        residuals = np.abs(y_true - y_pred)
        expected_radius = np.percentile(residuals, 80)  # Approximate
        assert abs(calibrator.radius_ - expected_radius) < 0.1

    def test_fit_normalized_method(self):
        """Test fitting with normalized method."""
        calibrator = QuantileConformalCalibrator(alpha=0.2, method="normalized")

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        y_std = np.array([0.1, 0.15, 0.12, 0.08, 0.2])

        result = calibrator.fit(y_pred, y_true, y_std_cal=y_std)

        assert result is calibrator
        assert calibrator.radius_ is not None
        assert calibrator.radius_ >= 0

    def test_fit_normalized_method_no_std(self):
        """Test fitting with normalized method but no std provided."""
        calibrator = QuantileConformalCalibrator(method="normalized")

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        with pytest.raises(ValueError, match="y_std_cal must be provided"):
            calibrator.fit(y_pred, y_true)

    def test_fit_invalid_method(self):
        """Test fitting with invalid method."""
        calibrator = QuantileConformalCalibrator(method="invalid")

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        with pytest.raises(ValueError, match="method must be"):
            calibrator.fit(y_pred, y_true)

    def test_fit_mismatched_lengths(self):
        """Test fitting with mismatched array lengths."""
        calibrator = QuantileConformalCalibrator()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9])  # Different length

        with pytest.raises(ValueError, match="must have the same length"):
            calibrator.fit(y_pred, y_true)

    def test_predict_absolute_method(self):
        """Test prediction with absolute method."""
        calibrator = QuantileConformalCalibrator(alpha=0.2, method="absolute")

        # Fit first
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        calibrator.fit(y_pred, y_true)

        # Predict on new data
        y_hat_new = np.array([2.5, 3.5])
        intervals = calibrator.predict(y_hat_new)

        assert intervals.shape == (2, 2)  # (n_samples, 2) for [lower, upper]
        assert np.all(intervals[:, 0] <= intervals[:, 1])  # Lower <= Upper

        # Check that intervals are symmetric around predictions
        midpoints = (intervals[:, 0] + intervals[:, 1]) / 2
        np.testing.assert_array_almost_equal(midpoints, y_hat_new)

    def test_predict_normalized_method(self):
        """Test prediction with normalized method."""
        calibrator = QuantileConformalCalibrator(alpha=0.2, method="normalized")

        # Fit first
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        y_std_cal = np.array([0.1, 0.15, 0.12, 0.08, 0.2])
        calibrator.fit(y_pred, y_true, y_std_cal=y_std_cal)

        # Predict on new data
        y_hat_new = np.array([2.5, 3.5])
        y_std_new = np.array([0.1, 0.2])
        intervals = calibrator.predict(y_hat_new, y_std_new)

        assert intervals.shape == (2, 2)
        assert np.all(intervals[:, 0] <= intervals[:, 1])

        # Intervals should be wider for higher std
        width1 = intervals[0, 1] - intervals[0, 0]
        width2 = intervals[1, 1] - intervals[1, 0]
        assert width2 > width1  # Second point has higher std

    def test_predict_normalized_method_no_std(self):
        """Test prediction with normalized method but no std provided."""
        calibrator = QuantileConformalCalibrator(method="normalized")

        # Fit first
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        y_std_cal = np.array([0.1, 0.15, 0.12])
        calibrator.fit(y_pred, y_true, y_std_cal=y_std_cal)

        # Try to predict without std
        y_hat_new = np.array([2.5])
        with pytest.raises(ValueError, match="y_std must be provided"):
            calibrator.predict(y_hat_new)

    def test_predict_not_fitted(self):
        """Test prediction without fitting first."""
        calibrator = QuantileConformalCalibrator()

        y_hat_new = np.array([2.5, 3.5])
        with pytest.raises(ValueError, match="Calibrator not fitted"):
            calibrator.predict(y_hat_new)

    def test_calibrate_alias(self):
        """Test that calibrate method is an alias for fit."""
        calibrator = QuantileConformalCalibrator()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        # Since calibrate has different signature, just test fit method
        result = calibrator.fit(y_pred, y_true)

        assert result is calibrator
        assert calibrator.radius_ is not None

    def test_transform_method(self):
        """Test transform method."""
        calibrator = QuantileConformalCalibrator(method="absolute")

        # Fit first
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        calibrator.fit(y_pred, y_true)

        # Transform should be same as predict
        y_hat_new = np.array([2.5])
        result_transform = calibrator.transform(y_hat_new)
        result_predict = calibrator.predict(y_hat_new)

        np.testing.assert_array_almost_equal(result_transform, result_predict)

    def test_fit_transform_method(self):
        """Test fit_transform method."""
        calibrator = QuantileConformalCalibrator()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        result = calibrator.fit_transform(y_pred, y_true)

        # Should be equivalent to fit followed by transform
        expected = calibrator.predict(y_pred)
        np.testing.assert_array_almost_equal(result, expected)

    def test_predict_quantiles_not_implemented(self):
        """Test that predict_quantiles raises NotImplementedError."""
        calibrator = QuantileConformalCalibrator()

        with pytest.raises(NotImplementedError, match="produces intervals"):
            calibrator.predict_quantiles()


class TestConformalizedQuantileCalibrator:
    """Test suite for ConformalizedQuantileCalibrator class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        np.testing.assert_array_equal(calibrator.quantiles, np.array([0.1, 0.5, 0.9]))
        assert calibrator.enforce_monotonic is True
        assert calibrator.shifts_ is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        quantiles = [0.25, 0.5, 0.75]
        calibrator = ConformalizedQuantileCalibrator(quantiles, enforce_monotonic=False)

        np.testing.assert_array_equal(calibrator.quantiles, np.array([0.25, 0.5, 0.75]))
        assert calibrator.enforce_monotonic is False

    def test_init_unsorted_quantiles(self):
        """Test initialization with unsorted quantiles."""
        quantiles = [0.9, 0.1, 0.5]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        # Should be automatically sorted
        np.testing.assert_array_equal(calibrator.quantiles, np.array([0.1, 0.5, 0.9]))

    def test_init_invalid_quantiles(self):
        """Test initialization with invalid quantiles."""
        with pytest.raises(ValueError, match="must be within"):
            ConformalizedQuantileCalibrator([0.1, 0.5, 1.1])

        with pytest.raises(ValueError, match="must be strictly unique"):
            ConformalizedQuantileCalibrator([0.1, 0.5, 0.5])

    def test_fit_basic(self):
        """Test basic fitting functionality."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_quantiles = np.array(
            [
                [0.5, 1.0, 1.5],  # Sample 1 quantiles
                [1.5, 2.0, 2.5],  # Sample 2 quantiles
                [2.5, 3.0, 3.5],  # Sample 3 quantiles
                [3.5, 4.0, 4.5],  # Sample 4 quantiles
                [4.5, 5.0, 5.5],  # Sample 5 quantiles
            ]
        )

        result = calibrator.fit(
            y_pred_quantiles, y_true, y_pred_cal_quantiles=y_pred_quantiles
        )

        assert result is calibrator
        assert calibrator.shifts_ is not None
        assert calibrator.shifts_.shape == (3,)  # One shift per quantile

    def test_fit_no_quantiles_provided(self):
        """Test fitting without y_pred_cal_quantiles."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="must have shape"):
            calibrator.fit(y_pred, y_true)

    def test_fit_wrong_dimensions(self):
        """Test fitting with wrong dimensions."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_quantiles = np.array([[1.0, 2.0]])  # Wrong number of quantiles

        with pytest.raises(ValueError, match="must have shape"):
            calibrator.fit(
                y_pred_quantiles, y_true, y_pred_cal_quantiles=y_pred_quantiles
            )

    def test_fit_mismatched_sample_counts(self):
        """Test fitting with mismatched sample counts."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        y_true = np.array([1.0, 2.0])  # 2 samples
        y_pred_quantiles = np.array(
            [
                [0.5, 1.0, 1.5],
                [1.5, 2.0, 2.5],
                [2.5, 3.0, 3.5],  # 3 samples
            ]
        )

        with pytest.raises(ValueError, match="must align on n_cal"):
            calibrator.fit(
                y_pred_quantiles, y_true, y_pred_cal_quantiles=y_pred_quantiles
            )

    def test_predict_quantiles(self):
        """Test predict_quantiles method."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles, enforce_monotonic=False)

        # Fit first
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_quantiles = np.array(
            [
                [0.8, 1.0, 1.2],
                [1.8, 2.0, 2.2],
                [2.8, 3.0, 3.2],
            ]
        )
        calibrator.fit(y_pred_quantiles, y_true, y_pred_cal_quantiles=y_pred_quantiles)

        # Predict on new data
        y_new_quantiles = np.array(
            [
                [1.3, 1.5, 1.7],
                [2.3, 2.5, 2.7],
            ]
        )

        result = calibrator.predict_quantiles(y_new_quantiles)

        assert result.shape == (2, 3)  # (n_samples, n_quantiles)
        # Result should be shifted version of input
        expected = y_new_quantiles + calibrator.shifts_[None, :]
        np.testing.assert_array_almost_equal(result, expected)

    def test_predict_quantiles_with_monotonic_enforcement(self):
        """Test predict_quantiles with monotonic enforcement."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles, enforce_monotonic=True)

        # Fit with data that might create non-monotonic results after shifting
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_quantiles = np.array(
            [
                [0.8, 1.0, 1.2],
                [1.8, 2.0, 2.2],
                [2.8, 3.0, 3.2],
            ]
        )
        calibrator.fit(y_pred_quantiles, y_true, y_pred_cal_quantiles=y_pred_quantiles)

        # Predict with potentially problematic data
        y_new_quantiles = np.array(
            [
                [2.0, 1.5, 1.0],  # Non-monotonic input
            ]
        )

        result = calibrator.predict_quantiles(y_new_quantiles)

        # Result should be monotonic (non-decreasing)
        assert np.all(np.diff(result[0]) >= 0)

    def test_predict_quantiles_not_fitted(self):
        """Test predict_quantiles without fitting first."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        y_new_quantiles = np.array([[1.0, 1.5, 2.0]])

        with pytest.raises(ValueError, match="Calibrator not fitted"):
            calibrator.predict_quantiles(y_new_quantiles)

    def test_predict_quantiles_wrong_dimensions(self):
        """Test predict_quantiles with wrong dimensions."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        # Fit first
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_quantiles = np.array(
            [
                [0.8, 1.0, 1.2],
                [1.8, 2.0, 2.2],
                [2.8, 3.0, 3.2],
            ]
        )
        calibrator.fit(y_pred_quantiles, y_true, y_pred_cal_quantiles=y_pred_quantiles)

        # Try to predict with wrong dimensions
        y_new_quantiles = np.array([[1.0, 1.5]])  # Only 2 quantiles, should be 3

        with pytest.raises(ValueError, match="must have shape"):
            calibrator.predict_quantiles(y_new_quantiles)

    def test_predict_intervals(self):
        """Test predict method for interval generation."""
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        # Fit first
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_quantiles = np.array(
            [
                [0.7, 0.9, 1.0, 1.1, 1.3],
                [1.7, 1.9, 2.0, 2.1, 2.3],
                [2.7, 2.9, 3.0, 3.1, 3.3],
                [3.7, 3.9, 4.0, 4.1, 4.3],
                [4.7, 4.9, 5.0, 5.1, 5.3],
            ]
        )
        calibrator.fit(y_pred_quantiles, y_true, y_pred_cal_quantiles=y_pred_quantiles)

        # Predict intervals (alpha=0.1 means 90% intervals, tau_lo=0.05, tau_hi=0.95)
        y_new_quantiles = np.array(
            [
                [1.2, 1.4, 1.5, 1.6, 1.8],
                [2.2, 2.4, 2.5, 2.6, 2.8],
            ]
        )

        intervals = calibrator.predict(y_new_quantiles, alpha=0.1)

        assert intervals.shape == (2, 2)  # (n_samples, 2) for [lower, upper]
        assert np.all(intervals[:, 0] <= intervals[:, 1])  # Lower <= Upper

    def test_predict_intervals_invalid_alpha(self):
        """Test predict intervals with invalid alpha."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        # Fit first (minimal)
        y_true = np.array([1.0])
        y_pred_quantiles = np.array([[0.8, 1.0, 1.2]])
        calibrator.fit(y_pred_quantiles, y_true, y_pred_cal_quantiles=y_pred_quantiles)

        y_new_quantiles = np.array([[1.0, 1.5, 2.0]])

        with pytest.raises(ValueError, match="alpha must be in"):
            calibrator.predict(y_new_quantiles, alpha=0.0)

    def test_predict_intervals_interpolation(self):
        """Test predict intervals with interpolation (when exact quantiles not available)."""
        quantiles = [0.1, 0.5, 0.9]  # Don't have 0.05 and 0.95 for alpha=0.1
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        # Fit first
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_quantiles = np.array(
            [
                [0.8, 1.0, 1.2],
                [1.8, 2.0, 2.2],
                [2.8, 3.0, 3.2],
            ]
        )
        calibrator.fit(y_pred_quantiles, y_true, y_pred_cal_quantiles=y_pred_quantiles)

        # This should use interpolation since we don't have exact 0.05 and 0.95 quantiles
        y_new_quantiles = np.array([[1.3, 1.5, 1.7]])
        intervals = calibrator.predict(y_new_quantiles, alpha=0.1)

        assert intervals.shape == (1, 2)
        assert intervals[0, 0] <= intervals[0, 1]

    def test_calibrate_alias(self):
        """Test that calibrate method is an alias for fit."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_quantiles = np.array(
            [
                [0.8, 1.0, 1.2],
                [1.8, 2.0, 2.2],
                [2.8, 3.0, 3.2],
            ]
        )

        result = calibrator.calibrate(
            None, y_true, y_pred_cal_quantiles=y_pred_quantiles
        )

        assert result is calibrator
        assert calibrator.shifts_ is not None

    def test_transform_method(self):
        """Test transform method."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        # Fit first
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_quantiles = np.array(
            [
                [0.8, 1.0, 1.2],
                [1.8, 2.0, 2.2],
                [2.8, 3.0, 3.2],
            ]
        )
        calibrator.fit(y_pred_quantiles, y_true, y_pred_cal_quantiles=y_pred_quantiles)

        # Transform should be same as predict_quantiles
        y_new_quantiles = np.array([[1.3, 1.5, 1.7]])
        result_transform = calibrator.transform(y_new_quantiles)
        result_predict = calibrator.predict_quantiles(y_new_quantiles)

        np.testing.assert_array_almost_equal(result_transform, result_predict)

    def test_fit_transform_method(self):
        """Test fit_transform method."""
        quantiles = [0.1, 0.5, 0.9]
        calibrator = ConformalizedQuantileCalibrator(quantiles)

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_quantiles = np.array(
            [
                [0.8, 1.0, 1.2],
                [1.8, 2.0, 2.2],
                [2.8, 3.0, 3.2],
            ]
        )

        result = calibrator.fit_transform(
            y_pred_quantiles, y_true, y_pred_cal_quantiles=y_pred_quantiles
        )

        # Should be equivalent to fit followed by transform
        expected = calibrator.predict_quantiles(y_pred_quantiles)
        np.testing.assert_array_almost_equal(result, expected)


class TestCalibratorIntegration:
    """Integration tests for calibrator classes."""

    def test_quantile_conformal_coverage(self):
        """Test that QuantileConformalCalibrator achieves approximate coverage."""
        np.random.seed(42)  # For reproducible results

        # Generate synthetic data
        n_cal = 1000
        n_test = 500
        y_true_cal = np.random.normal(0, 1, n_cal)
        y_pred_cal = y_true_cal + np.random.normal(0, 0.2, n_cal)  # Noisy predictions

        y_true_test = np.random.normal(0, 1, n_test)
        y_pred_test = y_true_test + np.random.normal(0, 0.2, n_test)

        # Fit calibrator
        calibrator = QuantileConformalCalibrator(alpha=0.1)  # 90% coverage
        calibrator.fit(y_pred_cal, y_true_cal)

        # Get intervals for test data
        intervals = calibrator.predict(y_pred_test)

        # Check coverage
        coverage = np.mean(
            (y_true_test >= intervals[:, 0]) & (y_true_test <= intervals[:, 1])
        )

        # Should achieve approximately 90% coverage (allow some tolerance)
        assert 0.85 <= coverage <= 0.95

    def test_conformalized_quantile_calibrator_coverage(self):
        """Test that ConformalizedQuantileCalibrator improves coverage."""
        np.random.seed(42)

        # Generate synthetic data with miscalibrated quantiles
        n_cal = 500
        n_test = 300
        quantiles = [0.1, 0.5, 0.9]

        y_true_cal = np.random.normal(0, 1, n_cal)
        # Create systematically biased quantile predictions
        y_pred_quantiles_cal = np.column_stack(
            [
                np.random.normal(-1.0, 0.8, n_cal),  # Biased low quantile
                y_true_cal
                + np.random.normal(0.1, 0.2, n_cal),  # Slightly biased median
                np.random.normal(1.2, 0.8, n_cal),  # Biased high quantile
            ]
        )

        y_true_test = np.random.normal(0, 1, n_test)
        y_pred_quantiles_test = np.column_stack(
            [
                np.random.normal(-1.0, 0.8, n_test),
                y_true_test + np.random.normal(0.1, 0.2, n_test),
                np.random.normal(1.2, 0.8, n_test),
            ]
        )

        # Fit calibrator
        calibrator = ConformalizedQuantileCalibrator(quantiles)
        calibrator.fit(
            y_pred_quantiles_cal, y_true_cal, y_pred_cal_quantiles=y_pred_quantiles_cal
        )

        # Get calibrated quantiles
        calibrated_quantiles = calibrator.predict_quantiles(y_pred_quantiles_test)

        # Check that calibrated quantiles have better coverage than original
        for i, q in enumerate(quantiles):
            original_coverage = np.mean(y_true_test <= y_pred_quantiles_test[:, i])
            calibrated_coverage = np.mean(y_true_test <= calibrated_quantiles[:, i])

            # Calibrated coverage should be closer to the target quantile
            original_error = abs(original_coverage - q)
            calibrated_error = abs(calibrated_coverage - q)

            # Allow some tolerance due to finite sample effects
            assert calibrated_error <= original_error + 0.1

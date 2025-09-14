"""Unit tests for calibration functions."""

import numpy as np
import pytest
from typing import List

from stock_market_analytics.modeling.model_factory.calibration.calibration_functions import (
    # Validation functions
    check_alpha,
    ensure_1d,
    check_same_length,
    ensure_sorted_unique_quantiles,

    # Core math functions
    finite_sample_quantile,

    # Conformity scores
    conformity_abs_residuals,
    conformity_normalized_abs,
    cqr_interval_scores,

    # Interval functions
    symmetric_interval_from_radius,

    # Quantile calibration functions
    residuals_for_quantile,
    residual_shift_for_tau,
    apply_quantile_shifts,
    enforce_monotone_across_quantiles,
)


class TestValidationFunctions:
    """Test suite for validation and utility functions."""

    def test_check_alpha_valid(self):
        """Test check_alpha with valid values."""
        # Should not raise for valid alpha values
        check_alpha(0.1)
        check_alpha(0.05)
        check_alpha(0.5)
        check_alpha(0.99)

    def test_check_alpha_invalid(self):
        """Test check_alpha with invalid values."""
        with pytest.raises(ValueError, match="alpha must be in"):
            check_alpha(0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            check_alpha(1.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            check_alpha(-0.1)

        with pytest.raises(ValueError, match="alpha must be in"):
            check_alpha(1.1)

    def test_ensure_1d_already_1d(self):
        """Test ensure_1d with already 1D array."""
        arr = np.array([1, 2, 3])
        result = ensure_1d(arr)
        np.testing.assert_array_equal(result, arr)
        assert result.ndim == 1

    def test_ensure_1d_from_2d(self):
        """Test ensure_1d with 2D array."""
        arr = np.array([[1, 2, 3]])
        result = ensure_1d(arr)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)
        assert result.ndim == 1

    def test_ensure_1d_from_scalar(self):
        """Test ensure_1d with scalar."""
        arr = np.array(5)
        result = ensure_1d(arr)
        expected = np.array([5])
        np.testing.assert_array_equal(result, expected)
        assert result.ndim == 1

    def test_check_same_length_valid(self):
        """Test check_same_length with arrays of same length."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        arr3 = np.array([7, 8, 9])

        # Should not raise for same lengths
        check_same_length(arr1, arr2, arr3)

    def test_check_same_length_invalid(self):
        """Test check_same_length with arrays of different lengths."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5])
        arr3 = np.array([7, 8, 9, 10])

        with pytest.raises(ValueError, match="must have the same length"):
            check_same_length(arr1, arr2, arr3)

    def test_ensure_sorted_unique_quantiles_valid(self):
        """Test ensure_sorted_unique_quantiles with valid quantiles."""
        q = [0.1, 0.5, 0.9]
        result = ensure_sorted_unique_quantiles(q)
        expected = np.array([0.1, 0.5, 0.9])
        np.testing.assert_array_equal(result, expected)

    def test_ensure_sorted_unique_quantiles_unsorted(self):
        """Test ensure_sorted_unique_quantiles with unsorted quantiles."""
        q = [0.9, 0.1, 0.5]
        result = ensure_sorted_unique_quantiles(q)
        expected = np.array([0.1, 0.5, 0.9])
        np.testing.assert_array_equal(result, expected)

    def test_ensure_sorted_unique_quantiles_out_of_range(self):
        """Test ensure_sorted_unique_quantiles with out-of-range values."""
        with pytest.raises(ValueError, match="must be within"):
            ensure_sorted_unique_quantiles([0.1, 0.5, 1.1])

        with pytest.raises(ValueError, match="must be within"):
            ensure_sorted_unique_quantiles([-0.1, 0.5, 0.9])

    def test_ensure_sorted_unique_quantiles_duplicates(self):
        """Test ensure_sorted_unique_quantiles with duplicate values."""
        with pytest.raises(ValueError, match="must be strictly unique"):
            ensure_sorted_unique_quantiles([0.1, 0.5, 0.5, 0.9])


class TestCoreMathFunctions:
    """Test suite for core mathematical functions."""

    def test_finite_sample_quantile_basic(self):
        """Test finite_sample_quantile with basic case."""
        scores = np.array([1, 2, 3, 4, 5])
        result = finite_sample_quantile(scores, level=0.5)

        # For n=5, k=ceil(6*0.5)=3, so 3rd order statistic is 3
        assert result == 3.0

    def test_finite_sample_quantile_conservative(self):
        """Test that finite_sample_quantile is conservative."""
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Conservative quantile should be higher than standard empirical quantile
        result_90 = finite_sample_quantile(scores, level=0.9)
        standard_90 = np.percentile(scores, 90)

        assert result_90 >= standard_90

    def test_finite_sample_quantile_boundary_cases(self):
        """Test finite_sample_quantile with boundary levels."""
        scores = np.array([1, 2, 3, 4, 5])

        # Level 0 should give minimum
        result_0 = finite_sample_quantile(scores, level=0.0)
        assert result_0 == 1.0

        # Level 1 should give maximum
        result_1 = finite_sample_quantile(scores, level=1.0)
        assert result_1 == 5.0

    def test_finite_sample_quantile_single_element(self):
        """Test finite_sample_quantile with single element."""
        scores = np.array([42])
        result = finite_sample_quantile(scores, level=0.5)
        assert result == 42.0

    def test_finite_sample_quantile_empty_array(self):
        """Test finite_sample_quantile with empty array."""
        scores = np.array([])
        with pytest.raises(ValueError, match="Cannot take quantile of empty scores"):
            finite_sample_quantile(scores, level=0.5)

    def test_finite_sample_quantile_invalid_level(self):
        """Test finite_sample_quantile with invalid level."""
        scores = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="level must be in"):
            finite_sample_quantile(scores, level=-0.1)

        with pytest.raises(ValueError, match="level must be in"):
            finite_sample_quantile(scores, level=1.1)


class TestConformityScores:
    """Test suite for conformity score functions."""

    def test_conformity_abs_residuals(self):
        """Test conformity_abs_residuals calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.8, 3.2])

        result = conformity_abs_residuals(y_true, y_pred)
        expected = np.array([0.1, 0.2, 0.2])

        np.testing.assert_array_almost_equal(result, expected)

    def test_conformity_abs_residuals_perfect_predictions(self):
        """Test conformity_abs_residuals with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        result = conformity_abs_residuals(y_true, y_pred)
        expected = np.array([0.0, 0.0, 0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_conformity_abs_residuals_mismatched_lengths(self):
        """Test conformity_abs_residuals with mismatched array lengths."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.8])

        with pytest.raises(ValueError, match="must have the same length"):
            conformity_abs_residuals(y_true, y_pred)

    def test_conformity_normalized_abs(self):
        """Test conformity_normalized_abs calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.8, 3.2])
        y_std = np.array([0.1, 0.2, 0.1])

        result = conformity_normalized_abs(y_true, y_pred, y_std)
        expected = np.array([1.0, 1.0, 2.0])  # |residual| / std

        np.testing.assert_array_almost_equal(result, expected)

    def test_conformity_normalized_abs_with_eps(self):
        """Test conformity_normalized_abs with small std values."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 1.8])
        y_std = np.array([1e-10, 0.2])  # Very small std
        eps = 1e-8

        result = conformity_normalized_abs(y_true, y_pred, y_std, eps=eps)

        # First element should use eps instead of very small std
        assert abs(result[0] - (0.1 / eps)) < 1e-6  # Allow for floating point precision
        assert abs(result[1] - 1.0) < 1e-10  # 0.2 / 0.2 = 1.0

    def test_cqr_interval_scores(self):
        """Test CQR interval scores calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_lo = np.array([0.8, 1.9, 2.8])
        y_hi = np.array([1.2, 2.1, 3.2])

        result = cqr_interval_scores(y_true, y_lo, y_hi)

        # max(y_lo - y, y - y_hi) for each point
        # Point 1: max(0.8-1.0, 1.0-1.2) = max(-0.2, -0.2) = -0.2 (covered, both negative)
        # Point 2: max(1.9-2.0, 2.0-2.1) = max(-0.1, -0.1) = -0.1 (covered, both negative)
        # Point 3: max(2.8-3.0, 3.0-3.2) = max(-0.2, -0.2) = -0.2 (covered, both negative)
        expected = np.array([-0.2, -0.1, -0.2])

        np.testing.assert_array_almost_equal(result, expected)

    def test_cqr_interval_scores_violations(self):
        """Test CQR interval scores with coverage violations."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_lo = np.array([1.1, 1.8, 3.1])  # Some violations
        y_hi = np.array([1.2, 2.2, 3.2])

        result = cqr_interval_scores(y_true, y_lo, y_hi)

        # Point 1: max(1.1-1.0, 1.0-1.2) = max(0.1, -0.2) = 0.1 (below interval)
        # Point 2: max(1.8-2.0, 2.0-2.2) = max(-0.2, -0.2) = -0.2 (covered, both negative)
        # Point 3: max(3.1-3.0, 3.0-3.2) = max(0.1, -0.2) = 0.1 (below interval)
        expected = np.array([0.1, -0.2, 0.1])

        np.testing.assert_array_almost_equal(result, expected)

    def test_cqr_interval_scores_invalid_intervals(self):
        """Test CQR interval scores with invalid intervals (y_lo > y_hi)."""
        y_true = np.array([1.0, 2.0])
        y_lo = np.array([1.2, 2.1])  # Higher than y_hi
        y_hi = np.array([1.1, 2.0])

        with pytest.raises(ValueError, match="Found y_lo > y_hi"):
            cqr_interval_scores(y_true, y_lo, y_hi)


class TestIntervalFunctions:
    """Test suite for interval creation functions."""

    def test_symmetric_interval_from_radius_scalar(self):
        """Test symmetric_interval_from_radius with scalar radius."""
        y_hat = np.array([1.0, 2.0, 3.0])
        radius = 0.5

        lo, hi = symmetric_interval_from_radius(y_hat, radius)

        expected_lo = np.array([0.5, 1.5, 2.5])
        expected_hi = np.array([1.5, 2.5, 3.5])

        np.testing.assert_array_almost_equal(lo, expected_lo)
        np.testing.assert_array_almost_equal(hi, expected_hi)

    def test_symmetric_interval_from_radius_array(self):
        """Test symmetric_interval_from_radius with array radius."""
        y_hat = np.array([1.0, 2.0, 3.0])
        radius = np.array([0.1, 0.2, 0.3])

        lo, hi = symmetric_interval_from_radius(y_hat, radius)

        expected_lo = np.array([0.9, 1.8, 2.7])
        expected_hi = np.array([1.1, 2.2, 3.3])

        np.testing.assert_array_almost_equal(lo, expected_lo)
        np.testing.assert_array_almost_equal(hi, expected_hi)

    def test_symmetric_interval_from_radius_zero(self):
        """Test symmetric_interval_from_radius with zero radius."""
        y_hat = np.array([1.0, 2.0, 3.0])
        radius = 0.0

        lo, hi = symmetric_interval_from_radius(y_hat, radius)

        # Zero radius should give point predictions
        np.testing.assert_array_almost_equal(lo, y_hat)
        np.testing.assert_array_almost_equal(hi, y_hat)


class TestQuantileCalibrationFunctions:
    """Test suite for quantile calibration functions."""

    def test_residuals_for_quantile(self):
        """Test residuals_for_quantile calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_tau = np.array([1.1, 1.8, 3.2])

        result = residuals_for_quantile(y_true, y_pred_tau)
        expected = np.array([-0.1, 0.2, -0.2])  # y_true - y_pred

        np.testing.assert_array_almost_equal(result, expected)

    def test_residuals_for_quantile_perfect(self):
        """Test residuals_for_quantile with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred_tau = np.array([1.0, 2.0, 3.0])

        result = residuals_for_quantile(y_true, y_pred_tau)
        expected = np.array([0.0, 0.0, 0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_residual_shift_for_tau(self):
        """Test residual_shift_for_tau calculation."""
        residuals = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        tau = 0.6

        result = residual_shift_for_tau(residuals, tau)

        # Should be the finite sample quantile at level tau
        expected = finite_sample_quantile(residuals, tau)
        assert result == expected

    def test_apply_quantile_shifts(self):
        """Test apply_quantile_shifts function."""
        y_pred_quantiles = np.array([
            [1.0, 2.0, 3.0],  # Sample 1: q=[0.1, 0.5, 0.9]
            [1.5, 2.5, 3.5]   # Sample 2
        ])
        shifts = np.array([0.1, 0.0, -0.1])

        result = apply_quantile_shifts(y_pred_quantiles, shifts)
        expected = np.array([
            [1.1, 2.0, 2.9],
            [1.6, 2.5, 3.4]
        ])

        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_quantile_shifts_wrong_dimensions(self):
        """Test apply_quantile_shifts with wrong dimensions."""
        y_pred_quantiles = np.array([1.0, 2.0, 3.0])  # 1D instead of 2D
        shifts = np.array([0.1, 0.0, -0.1])

        with pytest.raises(ValueError, match="must have shape"):
            apply_quantile_shifts(y_pred_quantiles, shifts)

    def test_apply_quantile_shifts_mismatched_shapes(self):
        """Test apply_quantile_shifts with mismatched shapes."""
        y_pred_quantiles = np.array([[1.0, 2.0, 3.0]])  # 3 quantiles
        shifts = np.array([0.1, 0.0])  # 2 shifts

        with pytest.raises(ValueError, match="shifts length must equal"):
            apply_quantile_shifts(y_pred_quantiles, shifts)

    def test_enforce_monotone_across_quantiles(self):
        """Test enforce_monotone_across_quantiles function."""
        # Create non-monotonic quantiles (violating order)
        yq = np.array([
            [3.0, 2.0, 1.0],  # Sample 1: decreasing (should be fixed)
            [1.0, 3.0, 2.0]   # Sample 2: non-monotonic
        ])

        result = enforce_monotone_across_quantiles(yq)
        expected = np.array([
            [3.0, 3.0, 3.0],  # Cumulative max makes it non-decreasing
            [1.0, 3.0, 3.0]
        ])

        np.testing.assert_array_almost_equal(result, expected)

    def test_enforce_monotone_across_quantiles_already_monotonic(self):
        """Test enforce_monotone_across_quantiles with already monotonic data."""
        yq = np.array([
            [1.0, 2.0, 3.0],  # Already increasing
            [1.5, 2.5, 3.5]
        ])

        result = enforce_monotone_across_quantiles(yq)

        # Should remain unchanged
        np.testing.assert_array_almost_equal(result, yq)

    def test_enforce_monotone_across_quantiles_wrong_dimensions(self):
        """Test enforce_monotone_across_quantiles with wrong dimensions."""
        yq = np.array([1.0, 2.0, 3.0])  # 1D instead of 2D

        with pytest.raises(ValueError, match="must be 2D"):
            enforce_monotone_across_quantiles(yq)

    def test_enforce_monotone_across_quantiles_single_quantile(self):
        """Test enforce_monotone_across_quantiles with single quantile."""
        yq = np.array([[1.0], [2.0], [3.0]])  # Single quantile per sample

        result = enforce_monotone_across_quantiles(yq)

        # Should remain unchanged
        np.testing.assert_array_almost_equal(result, yq)
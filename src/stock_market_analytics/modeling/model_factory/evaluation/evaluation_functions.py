"""
Helper functions for model evaluation.

This module provides mathematical functions and utilities needed to perform
various evaluation tasks for machine learning models in financial contexts.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

# -------------------------
# Core helpers (single task)
# -------------------------


def ensure_sorted_unique_quantiles(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    q_sorted = np.sort(q)
    if np.any((q_sorted < 0) | (q_sorted > 1)):
        raise ValueError("Quantiles must be in [0, 1].")
    if np.unique(q_sorted).shape[0] != q_sorted.shape[0]:
        raise ValueError("Quantiles must be strictly unique.")
    return q_sorted


def align_predictions_to_quantiles(
    y_pred_quantiles: np.ndarray, quantiles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort quantiles and reorder prediction columns to match.
    Returns (y_pred_aligned, q_sorted, order_idx)
    """
    y_pred_quantiles = np.asarray(y_pred_quantiles, dtype=float)
    q_sorted = ensure_sorted_unique_quantiles(quantiles)
    order_idx = np.argsort(quantiles)
    y_pred_aligned = y_pred_quantiles[:, order_idx]
    if y_pred_aligned.shape[1] != q_sorted.shape[0]:
        raise ValueError("n_quantiles mismatch between predictions and quantiles.")
    return y_pred_aligned, q_sorted, order_idx


def validate_xyq_shapes(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: np.ndarray
) -> None:
    y_true = np.asarray(y_true).reshape(-1)
    if y_pred_quantiles.ndim != 2:
        raise ValueError("y_pred_quantiles must be 2D (n_samples, n_quantiles).")
    if y_true.shape[0] != y_pred_quantiles.shape[0]:
        raise ValueError("y_true length must equal n_samples of y_pred_quantiles.")


def drop_nan_rows_for_quantiles(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Remove rows with any NaNs in y_true or predicted quantiles.
    Returns (y_true_clean, yq_clean, n_dropped)
    """
    y_true = np.asarray(y_true).reshape(-1)
    yq = np.asarray(y_pred_quantiles, dtype=float)
    if yq.ndim != 2:
        raise ValueError("y_pred_quantiles must be 2D.")
    mask = ~np.isnan(y_true)
    mask &= ~np.any(np.isnan(yq), axis=1)
    n_dropped = int(y_true.shape[0] - np.sum(mask))
    return y_true[mask], yq[mask, :], n_dropped


def enforce_monotone_across_quantiles(yq: np.ndarray) -> np.ndarray:
    """Make each row nondecreasing along τ via cumulative max (evaluation-only repair)."""
    yq = np.asarray(yq, dtype=float)
    if yq.ndim != 2:
        raise ValueError("yq must be 2D.")
    return np.maximum.accumulate(yq, axis=1)


def monotonicity_violation_rate(y_pred_quantiles: np.ndarray) -> Tuple[float, int]:
    """
    Fraction and count of rows where any q_{k+1} < q_k (i.e., diff < 0).
    """
    if y_pred_quantiles.shape[1] <= 1:
        return 0.0, 0
    diffs = np.diff(y_pred_quantiles, axis=1)
    violated_rows = np.any(diffs < 0, axis=1)
    count = int(np.sum(violated_rows))
    rate = (
        float(count / y_pred_quantiles.shape[0]) if y_pred_quantiles.shape[0] else 0.0
    )
    return rate, count


def pinball_loss_vectorized(
    y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float).reshape(-1)
    validate_xyq_shapes(y_true, y_pred, quantiles)
    e = y_true[:, None] - y_pred  # (n, q)
    q = quantiles[None, :]  # (1, q)
    losses = np.maximum(q * e, (q - 1.0) * e)
    return np.mean(losses, axis=0)  # per-quantile


def quantile_coverage(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_quantiles = np.asarray(y_pred_quantiles, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float).reshape(-1)
    validate_xyq_shapes(y_true, y_pred_quantiles, quantiles)
    return np.mean(y_true[:, None] <= y_pred_quantiles, axis=0)


def quantile_loss_differential(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: np.ndarray
) -> Dict[str, float]:
    pinball_losses = pinball_loss_vectorized(y_true, y_pred_quantiles, quantiles)
    cov = quantile_coverage(y_true, y_pred_quantiles, quantiles)
    cov_err = cov - quantiles
    return {
        "mean_pinball_loss": float(np.mean(pinball_losses)),
        "max_pinball_loss": float(np.max(pinball_losses)),
        "mean_coverage_error": float(np.mean(np.abs(cov_err))),
        "max_coverage_error": float(np.max(np.abs(cov_err))),
        "coverage_bias": float(np.mean(cov_err)),  # <0 under-coverage
    }


def crps_from_quantiles(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: np.ndarray
) -> float:
    """
    Approximate CRPS via 2 * integral_0^1 pinball(y, Q_tau) d tau, using trapezoidal rule over provided taus.
    Accurate when the tau grid is reasonably dense and spans near 0 and 1.
    """
    q = ensure_sorted_unique_quantiles(quantiles)
    y_pred_sorted, _, _ = align_predictions_to_quantiles(y_pred_quantiles, q)
    pinball_per_tau = pinball_loss_vectorized(
        y_true, y_pred_sorted, q
    )  # (n_quantiles,)
    area = np.trapezoid(pinball_per_tau, q)
    return float(2.0 * area)


def interval_score(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, alpha: float = 0.1
) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    y_true = np.asarray(y_true).reshape(-1)
    y_lower = np.asarray(y_lower).reshape(-1)
    y_upper = np.asarray(y_upper).reshape(-1)
    if not (y_true.shape == y_lower.shape == y_upper.shape):
        raise ValueError("y_true, y_lower, y_upper must have the same shape.")
    if np.any(y_lower > y_upper):
        raise ValueError("Found y_lower > y_upper.")
    width = y_upper - y_lower
    below = y_true < y_lower
    above = y_true > y_upper
    lower_penalty = (2.0 / alpha) * (y_lower - y_true) * below
    upper_penalty = (2.0 / alpha) * (y_true - y_upper) * above
    total = width + lower_penalty + upper_penalty
    return float(np.mean(total))


def prediction_interval_coverage_probability(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray
) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_lower = np.asarray(y_lower).reshape(-1)
    y_upper = np.asarray(y_upper).reshape(-1)
    if not (y_true.shape == y_lower.shape == y_upper.shape):
        raise ValueError("y_true, y_lower, y_upper must have the same shape.")
    if np.any(y_lower > y_upper):
        raise ValueError("Found y_lower > y_upper.")
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(covered))


def mean_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    y_lower = np.asarray(y_lower).reshape(-1)
    y_upper = np.asarray(y_upper).reshape(-1)
    if y_lower.shape != y_upper.shape:
        raise ValueError("y_lower and y_upper must have the same shape.")
    if np.any(y_lower > y_upper):
        raise ValueError("Found y_lower > y_upper.")
    return float(np.mean(y_upper - y_lower))


def normalized_interval_width(
    y_lower: np.ndarray, y_upper: np.ndarray, y_true: np.ndarray
) -> float:
    y_lower = np.asarray(y_lower).reshape(-1)
    y_upper = np.asarray(y_upper).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    if not (y_lower.shape == y_upper.shape == y_true.shape):
        raise ValueError("y_lower, y_upper, y_true must have the same shape.")
    if np.any(y_lower > y_upper):
        raise ValueError("Found y_lower > y_upper.")
    width = y_upper - y_lower
    denom = float(np.std(y_true))
    if not np.isfinite(denom) or denom <= 0.0:
        return 0.0 if np.allclose(width, 0.0) else float("inf")
    return float(np.mean(width) / denom)


def intervals_from_quantiles(
    y_pred_quantiles: np.ndarray, quantiles: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (lower, upper) for a (1-alpha) PI from a quantile matrix using linear interpolation if needed.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    q = ensure_sorted_unique_quantiles(quantiles)
    preds, _, _ = align_predictions_to_quantiles(y_pred_quantiles, q)
    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
    # exact columns?
    idx_lo = np.where(np.isclose(q, q_lo, atol=1e-12))[0]
    idx_hi = np.where(np.isclose(q, q_hi, atol=1e-12))[0]
    if idx_lo.size and idx_hi.size:
        return preds[:, idx_lo[0]], preds[:, idx_hi[0]]
    # otherwise interpolate row-wise (ensure monotone first for safety)
    preds_mono = enforce_monotone_across_quantiles(preds)
    lower = np.array([np.interp(q_lo, q, row) for row in preds_mono], dtype=float)
    upper = np.array([np.interp(q_hi, q, row) for row in preds_mono], dtype=float)
    return lower, upper


# ---------- PIT calibration helpers ----------


def pit_values(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """
    Probability Integral Transform using the quantile function:
    For each row, PIT = F_hat(y) ≈ interp of y over monotone Q_tau(x) vs tau grid.
    """
    q = ensure_sorted_unique_quantiles(quantiles)
    yq, _, _ = align_predictions_to_quantiles(y_pred_quantiles, q)
    yq = enforce_monotone_across_quantiles(yq)  # evaluation-only repair
    y_true = np.asarray(y_true).reshape(-1)
    # row-wise interpolation: tau = interp(y, Q_row, q), clipped to [0,1]
    pit = np.empty_like(y_true, dtype=float)
    for i, (y, row) in enumerate(zip(y_true, yq)):
        pit[i] = float(np.interp(y, row, q, left=0.0, right=1.0))
    return pit


def pit_ks_statistic(pit: np.ndarray) -> float:
    """
    One-sample Kolmogorov–Smirnov statistic vs Uniform(0,1) (no p-value; SciPy-free).
    """
    u = np.sort(np.asarray(pit, dtype=float).reshape(-1))
    n = u.size
    if n == 0:
        return float("nan")
    grid = np.arange(1, n + 1) / n
    d_plus = np.max(grid - u)
    d_minus = np.max(u - (np.arange(0, n) / n))
    return float(max(d_plus, d_minus))


def pit_ece(pit: np.ndarray, n_bins: int = 20) -> float:
    """
    Expected Calibration Error for PIT histogram (L1 deviation from uniform).
    """
    pit = np.asarray(pit, dtype=float).reshape(-1)
    if pit.size == 0:
        return float("nan")
    hist, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    freqs = hist / hist.sum()
    expected = 1.0 / n_bins
    return float(np.mean(np.abs(freqs - expected)))

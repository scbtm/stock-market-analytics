"""
Core evaluation functions for quantile regression models.

This module contains the mathematical functions used by evaluators to assess
model performance, including pinball loss, coverage metrics, and calibration error.
"""

from typing import Any

import numpy as np


def _weighted_mean(x: np.ndarray, w: np.ndarray | None) -> float:
    """Compute weighted mean of array x with optional weights w."""
    if w is None:
        return float(np.mean(x))
    
    w = np.asarray(w, dtype=float).ravel()
    w = w / np.sum(w)
    return float(np.sum(w * x))


def _pinball(
    y: np.ndarray, q: np.ndarray, alpha: float, w: np.ndarray | None = None
) -> float:
    """Compute pinball loss for quantile predictions."""
    e = y - q
    return _weighted_mean(np.maximum(alpha * e, (alpha - 1.0) * e), w)


def _interp(alpha: float, Q: list[float], qhat: np.ndarray) -> np.ndarray:
    """Interpolate quantile predictions for coverage/width calculation."""
    idx = np.where(np.isclose(Q, alpha))[0]
    if idx.size:
        return qhat[:, idx[0]]
    # linear interpolate between nearest quantiles
    if alpha <= Q[0] or alpha >= Q[-1]:
        raise ValueError(
            "interval alpha outside provided quantiles; add tails or change interval."
        )
    j_hi = np.searchsorted(Q, alpha)
    j_lo = j_hi - 1
    w_hi = (alpha - Q[j_lo]) / (Q[j_hi] - Q[j_lo])
    return (1.0 - w_hi) * qhat[:, j_lo] + w_hi * qhat[:, j_hi]


def eval_multiquantile(
    y_true: np.ndarray,
    q_pred: np.ndarray,  # shape (n_samples, n_quantiles), raw model output
    quantiles: list[float],  # e.g., [0.10, 0.25, 0.50, 0.75, 0.90] in the same column order as q_pred
    interval: tuple[float, float] = (0.10, 0.90),  # for coverage/width tracking
    sample_weight: np.ndarray | None = None,  # optional time-weights
    lambda_cross: float = 0.0,  # >0 to discourage quantile crossing (penalty added to objective)
    return_per_quantile: bool = False,  # include per-quantile pinballs in metrics
) -> tuple[float, dict[str, Any]]:
    """
    Evaluate multi-quantile predictions with comprehensive metrics.
    
    Returns:
      loss: float  (scalar to minimize in Optuna)
      metrics: dict (coverage, mean_width, pinball_mean, etc.)

    Notes:
      - Uses raw q_pred (no sorting), so the crossing penalty measures real violations.
      - Pinball loss is a proper score for quantiles; averaging across quantiles is a solid single-number objective.
    """
    y = np.asarray(y_true).ravel()
    qhat = np.asarray(q_pred)
    Q = np.asarray(quantiles, dtype=float)

    assert qhat.ndim == 2 and qhat.shape[0] == y.shape[0], "Shape mismatch."
    assert len(Q) == qhat.shape[1], "quantiles must align with q_pred columns."
    assert 0.0 < interval[0] < interval[1] < 1.0, "interval must be within (0,1)."

    # --- pinball loss (objective base) ---
    pinballs = [
        _pinball(y, qhat[:, j], Q[j], sample_weight) for j in range(qhat.shape[1])
    ]
    pinball_mean = float(np.mean(pinballs))

    # --- optional crossing penalty (keeps it simple but nudges toward monotone) ---
    if lambda_cross > 0.0:
        diffs = qhat[:, :-1] - qhat[:, 1:]  # >0 means crossing
        cross_pen = _weighted_mean(np.clip(diffs, 0.0, None).sum(axis=1), sample_weight)
    else:
        cross_pen = 0.0

    loss = pinball_mean + lambda_cross * cross_pen

    q_lo = _interp(alpha=interval[0], Q=Q, qhat=qhat)
    q_hi = _interp(alpha=interval[1], Q=Q, qhat=qhat)
    covered = (y >= q_lo) & (y <= q_hi)

    coverage = _weighted_mean(covered.astype(float), sample_weight)
    mean_width = _weighted_mean(q_hi - q_lo, sample_weight)

    # --- quantile calibration mismatch (track-only) ---
    cal_errs = [
        abs(_weighted_mean((y <= qhat[:, j]).astype(float), sample_weight) - Q[j])
        for j in range(len(Q))
    ]
    cal_err_mean = float(np.mean(cal_errs))

    metrics = {
        "loss": loss,
        "pinball_mean": pinball_mean,
        f"coverage_{int(interval[0] * 100)}_{int(interval[1] * 100)}": coverage,
        "mean_width": mean_width,
        "crossing_penalty": cross_pen,
        "calibration_error_mean": cal_err_mean,
    }
    if return_per_quantile:
        for j, a in enumerate(Q):
            metrics[f"pinball@{a:.2f}"] = float(pinballs[j])

    return loss, metrics


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Calculate coverage of prediction intervals."""
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    """Calculate mean width of prediction intervals."""
    return float(np.mean(hi - lo))


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    """Calculate pinball loss for a single quantile."""
    e = y - q_pred
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))
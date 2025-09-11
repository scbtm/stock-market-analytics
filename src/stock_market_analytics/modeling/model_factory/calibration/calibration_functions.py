"""
Helper functions for calibration methods.

This module provides mathematical functions and utilities needed to perform
various calibration tasks on model predictions.
"""

from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple

# ---------- Validation ----------


def check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")


def ensure_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        a = a.reshape(-1)
    return a


def check_same_length(*arrays: Iterable[np.ndarray]) -> None:
    lengths = [len(ensure_1d(a)) for a in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All arrays must have the same length, got lengths={lengths}."
        )


def ensure_sorted_unique_quantiles(q: Iterable[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    if np.any((q < 0) | (q > 1)):
        raise ValueError("Quantiles must be within [0, 1].")
    uq = np.unique(q)
    if uq.size != q.size:
        raise ValueError("Quantiles must be strictly unique.")
    return np.sort(q)


# ---------- Core math ----------


def finite_sample_quantile(scores: np.ndarray, level: float) -> float:
    """
    Conservative empirical quantile: the k-th order statistic with
    k = ceil((n+1)*level). This matches split conformal's finite-sample guarantee.
    """
    scores = ensure_1d(np.asarray(scores, dtype=float))
    if scores.size == 0:
        raise ValueError("Cannot take quantile of empty scores.")
    if not (0.0 <= level <= 1.0):
        raise ValueError("level must be in [0, 1].")
    n = scores.size
    k = int(np.ceil((n + 1) * level))
    k = min(max(k, 1), n)  # clamp to [1, n]
    # kth order statistic (1-indexed) is element at index k-1 after partition
    return float(np.partition(scores, k - 1)[k - 1])


# ---------- Conformity scores ----------


def conformity_abs_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = ensure_1d(y_true)
    y_pred = ensure_1d(y_pred)
    check_same_length(y_true, y_pred)
    return np.abs(y_true - y_pred)


def conformity_normalized_abs(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    y_true = ensure_1d(y_true)
    y_pred = ensure_1d(y_pred)
    y_std = ensure_1d(y_std)
    check_same_length(y_true, y_pred, y_std)
    denom = np.maximum(y_std, eps)
    return np.abs(y_true - y_pred) / denom


def cqr_interval_scores(
    y_true: np.ndarray, y_lo: np.ndarray, y_hi: np.ndarray
) -> np.ndarray:
    """Romano et al. (2019): max( y_lo - y, y - y_hi ); nonnegative by def."""
    y_true = ensure_1d(y_true)
    y_lo = ensure_1d(y_lo)
    y_hi = ensure_1d(y_hi)
    check_same_length(y_true, y_lo, y_hi)
    if np.any(y_lo > y_hi):
        raise ValueError("Found y_lo > y_hi in CQR scores.")
    return np.maximum(y_lo - y_true, y_true - y_hi)


# ---------- Intervals ----------


def symmetric_interval_from_radius(
    y_hat: np.ndarray, radius: float | np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y_hat = ensure_1d(y_hat)
    r = np.asarray(radius, dtype=float)
    if r.ndim == 0:
        r = np.full_like(y_hat, float(r))
    check_same_length(y_hat, r)
    return y_hat - r, y_hat + r


# ---------- Quantile calibration (per τ) ----------


def residuals_for_quantile(y_true: np.ndarray, y_pred_tau: np.ndarray) -> np.ndarray:
    """Residuals for τ-quantile: r_i^τ = y_i - ŷ_τ(x_i)."""
    y_true = ensure_1d(y_true)
    y_pred_tau = ensure_1d(y_pred_tau)
    check_same_length(y_true, y_pred_tau)
    return y_true - y_pred_tau


def residual_shift_for_tau(residuals_tau: np.ndarray, tau: float) -> float:
    """
    Conservative shift ŝ_τ = empirical quantile of residuals at level τ
    using finite-sample 'higher' quantile. Calibrated quantile = ŷ_τ + ŝ_τ.
    """
    return finite_sample_quantile(residuals_tau, tau)


def apply_quantile_shifts(
    y_pred_quantiles: np.ndarray, shifts: np.ndarray
) -> np.ndarray:
    y_pred_quantiles = np.asarray(y_pred_quantiles, dtype=float)
    shifts = ensure_1d(shifts)
    if y_pred_quantiles.ndim != 2:
        raise ValueError("y_pred_quantiles must have shape (n_samples, n_quantiles).")
    if y_pred_quantiles.shape[1] != shifts.size:
        raise ValueError("shifts length must equal n_quantiles.")
    return y_pred_quantiles + shifts[None, :]


def enforce_monotone_across_quantiles(yq: np.ndarray) -> np.ndarray:
    """Make each row nondecreasing along τ via cumulative max (post-hoc repair)."""
    yq = np.asarray(yq, dtype=float)
    if yq.ndim != 2:
        raise ValueError("yq must be 2D.")
    return np.maximum.accumulate(yq, axis=1)

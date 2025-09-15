"""
Calibrator implementations for post-hoc prediction processing.

This module contains calibrator classes that implement the calibration protocols
and can be integrated into sklearn pipelines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from stock_market_analytics.modeling.model_factory.calibration.calibration_functions import (
    apply_quantile_shifts,
    check_alpha,
    check_same_length,
    conformity_abs_residuals,
    conformity_normalized_abs,
    enforce_monotone_across_quantiles,
    ensure_1d,
    ensure_sorted_unique_quantiles,
    finite_sample_quantile,
    residual_shift_for_tau,
    residuals_for_quantile,
    symmetric_interval_from_radius,
)
from stock_market_analytics.modeling.model_factory.protocols import (
    Array,
    BaseCalibrator,
    Frame,
    QuantileCalibrator,
    Series,
)


class QuantileConformalCalibrator(BaseEstimator, TransformerMixin, BaseCalibrator):
    """
    Split conformal calibrator for **prediction intervals** from a point predictor.
    Finite-sample marginal coverage 1 - alpha (under exchangeability).

    method="absolute": scores = |y - μ̂|
    method="normalized": scores = |y - μ̂| / σ̂  (requires y_std in fit & predict)
    """

    def __init__(self, alpha: float = 0.1, method: str = "absolute", eps: float = 1e-8):
        check_alpha(alpha)
        self.alpha = alpha
        self.method = method
        self.eps = eps
        self.radius_: float | None = None  # q_{1-alpha} of scores

    # sklearn-style alias requested in your stub
    def calibrate(
        self, X_cal: Frame | None, y_cal: Series, **kwargs: Any
    ) -> QuantileConformalCalibrator:
        return self.fit(X_cal, y_cal, **kwargs)

    def fit(
        self,
        y_pred_cal: Array,
        y_true_cal: Array,
        **kwargs: Any,
    ) -> QuantileConformalCalibrator:
        y_std_cal = kwargs.get("y_std_cal")
        y_true = np.asarray(y_true_cal).reshape(-1)
        y_pred_cal = np.asarray(y_pred_cal).reshape(-1)
        check_same_length(y_true, y_pred_cal)

        if self.method == "absolute":
            scores = conformity_abs_residuals(y_true, y_pred_cal)
        elif self.method == "normalized":
            if y_std_cal is None:
                raise ValueError("y_std_cal must be provided for method='normalized'.")
            scores = conformity_normalized_abs(
                y_true, y_pred_cal, y_std_cal, eps=self.eps
            )
        else:
            raise ValueError("method must be 'absolute' or 'normalized'.")

        self.radius_ = finite_sample_quantile(scores, level=1.0 - self.alpha)
        return self

    def predict(self, y_hat: np.ndarray, y_std: np.ndarray | None = None) -> np.ndarray:
        """
        Returns (n_samples, 2) array of [lower, upper].
        For method='normalized' you must pass y_std for the test points.
        """
        if self.radius_ is None:
            raise ValueError("Calibrator not fitted.")
        y_hat = ensure_1d(y_hat)

        if self.method == "absolute":
            lo, hi = symmetric_interval_from_radius(y_hat, self.radius_)
        else:  # normalized
            if y_std is None:
                raise ValueError("y_std must be provided for method='normalized'.")
            r = self.radius_ * ensure_1d(y_std)
            lo, hi = symmetric_interval_from_radius(y_hat, r)

        return np.column_stack((lo, hi))

    def transform(self, y_pred: Array, **kwargs: Any) -> Array:
        """Map raw model outputs to calibrated outputs."""
        y_std = kwargs.get("y_std")
        return self.predict(y_pred, y_std)

    def fit_transform(
        self,
        y_pred_cal: Array,
        y_true_cal: Array,
        **kwargs: Any,
    ) -> Array:
        """Fit the calibrator and transform calibration data."""
        self.fit(y_pred_cal, y_true_cal, **kwargs)
        return self.transform(y_pred_cal, **kwargs)

    # This class focuses on intervals; calibrated quantiles belong in the class below.
    def predict_quantiles(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError(
            "QuantileConformalCalibrator produces intervals. "
            "Use ConformalizedQuantileCalibrator for calibrated quantiles."
        )


class ConformalizedQuantileCalibrator(
    BaseEstimator, TransformerMixin, QuantileCalibrator
):
    """
    Per-τ **quantile calibration** via conservative residual shifts:
      r_i^τ = y_i - ŷ_τ(x_i),  ŝ_τ = Q̂_τ({r_i^τ}) with finite-sample 'higher' quantile,
      ŷ_τ^cal(x) = ŷ_τ(x) + ŝ_τ.
    Optionally enforces non-crossing across τ by a row-wise cumulative max.

    This produces calibrated **quantiles**; you can derive intervals by choosing τ_l=α/2, τ_u=1-α/2.
    """

    def __init__(self, quantiles: list[float], enforce_monotonic: bool = True):
        self.quantiles = ensure_sorted_unique_quantiles(quantiles)
        self.enforce_monotonic = enforce_monotonic
        self.shifts_: np.ndarray | None = None  # shape (n_quantiles,)

    def calibrate(
        self, X_cal: Frame | None, y_cal: Series, **kwargs: Any
    ) -> ConformalizedQuantileCalibrator:
        return self.fit(X_cal, y_cal, **kwargs)

    def fit(
        self,
        y_pred_cal: Array,
        y_true_cal: Array,
        **kwargs: Any,
    ) -> ConformalizedQuantileCalibrator:
        y_pred_cal_quantiles = kwargs.get("y_pred_cal_quantiles", y_pred_cal)
        if y_pred_cal_quantiles is None:
            raise ValueError(
                "y_pred_cal_quantiles must be provided with shape (n_cal, n_quantiles)."
            )
        y_true = np.asarray(y_true_cal).reshape(-1)
        Yq = np.asarray(y_pred_cal_quantiles, dtype=float)
        if Yq.ndim != 2 or Yq.shape[1] != self.quantiles.size:
            raise ValueError(
                "y_pred_cal_quantiles must have shape (n_cal, len(quantiles))."
            )
        if Yq.shape[0] != y_true.size:
            raise ValueError("y_cal and y_pred_cal_quantiles must align on n_cal.")

        # Compute per-τ conservative shift
        shifts = np.empty(self.quantiles.size, dtype=float)
        for j, tau in enumerate(self.quantiles):
            res_tau = residuals_for_quantile(y_true, Yq[:, j])
            shifts[j] = residual_shift_for_tau(res_tau, tau)
        self.shifts_ = shifts
        return self

    def predict_quantiles(self, y_pred_quantiles: np.ndarray) -> np.ndarray:
        """
        Apply stored per-τ shifts to a matrix of predicted quantiles (n_samples, n_quantiles).
        """
        if self.shifts_ is None:
            raise ValueError("Calibrator not fitted.")
        Yq = np.asarray(y_pred_quantiles, dtype=float)
        if Yq.ndim != 2 or Yq.shape[1] != self.quantiles.size:
            raise ValueError(
                "y_pred_quantiles must have shape (n_samples, len(quantiles))."
            )

        Yq_cal = apply_quantile_shifts(Yq, self.shifts_)
        if self.enforce_monotonic:
            Yq_cal = enforce_monotone_across_quantiles(Yq_cal)
        return Yq_cal

    def predict(self, y_pred_quantiles: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        check_alpha(alpha)
        Yq_cal = self.predict_quantiles(y_pred_quantiles)  # (n_samples, n_quantiles)
        q = self.quantiles
        tau_lo, tau_hi = alpha / 2.0, 1.0 - alpha / 2.0

        # exact-column fast path
        idx_lo = np.where(np.isclose(q, tau_lo, atol=1e-12))[0]
        idx_hi = np.where(np.isclose(q, tau_hi, atol=1e-12))[0]
        if idx_lo.size:
            lo = Yq_cal[:, idx_lo[0]]
        else:
            lo = np.array([np.interp(tau_lo, q, row) for row in Yq_cal], dtype=float)
        if idx_hi.size:
            hi = Yq_cal[:, idx_hi[0]]
        else:
            hi = np.array([np.interp(tau_hi, q, row) for row in Yq_cal], dtype=float)

        return np.column_stack((lo, hi))

    def transform(self, y_pred_quantiles: Array) -> Array:
        """Map raw quantile outputs to calibrated quantile outputs."""
        return self.predict_quantiles(y_pred_quantiles)

    def fit_transform(
        self,
        y_pred_cal: Array,
        y_true_cal: Array,
        **kwargs: Any,
    ) -> Array:
        """Fit the calibrator and transform calibration data."""
        self.fit(y_pred_cal, y_true_cal, **kwargs)
        return self.transform(y_pred_cal)

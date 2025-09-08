"""
Cross-cutting utility functions for pipeline components.

This module contains utility functions that are used across multiple domains
(evaluation, calibration, prediction) and don't belong to any single component.
"""

from typing import Any, Sequence
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from .protocols import (
    SupportsPredict,
    SupportsPredictProba, 
    SupportsPredictQuantiles,
    PointPreds,
    ProbPreds,
    QuantilePreds,
)

NDArrayF = npt.NDArray[np.float64]


def _to_1d(a: NDArrayF) -> NDArrayF:
    """Ravel to (n,), copying as float64."""
    return np.asarray(a, dtype=float).reshape(-1)


def extract_point(est: BaseEstimator, X: Any) -> PointPreds:
    """Extract point predictions from an estimator."""
    assert isinstance(est, SupportsPredict), (
        "Estimator does not implement predict(X)."
    )
    y_hat = _to_1d(np.asarray(est.predict(X)))
    return PointPreds(y_hat=y_hat)


def extract_proba(est: BaseEstimator, X: Any, classes: Sequence[Any] | None = None) -> ProbPreds:
    """Extract probability predictions from an estimator."""
    assert isinstance(est, SupportsPredictProba), (
        "Estimator does not implement predict_proba(X)."
    )
    P = np.asarray(est.predict_proba(X), dtype=float)
    return ProbPreds(proba=P, classes_=tuple(classes) if classes else tuple(range(P.shape[1])))


def extract_quantiles(
    est: BaseEstimator,
    X: Any,
    quantiles: Sequence[float] | NDArrayF | None = None,
) -> QuantilePreds:
    """Extract quantile predictions from an estimator."""
    assert isinstance(est, SupportsPredictQuantiles), (
        "Estimator does not implement predict_quantiles(X, quantiles=None)."
    )
    q = np.asarray(est.predict_quantiles(X, quantiles=quantiles), dtype=float)
    q_levels = tuple(quantiles) if quantiles is not None else tuple(range(q.shape[1]))
    return QuantilePreds(q=q, quantiles=q_levels)
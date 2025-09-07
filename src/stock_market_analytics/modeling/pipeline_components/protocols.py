from __future__ import annotations

"""
Best-in-class protocols for modular ML components.

Key ideas:
- Capability protocols express *what* an estimator can do (predict, predict_proba, predict_quantiles)
- Evaluators take ground truth + a typed PredictionBundle (not a model)
- Calibrators wrap an existing fitted estimator into a new sklearn-compatible estimator
- Everything remains sklearn-native (get_params/set_params where relevant)

This file is intentionally dependency-light: numpy, pandas, sklearn.base only.
"""

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
    TypeGuard,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator


# =========================
# Common task/type aliases
# =========================

TaskType = Literal["regression", "classification", "quantile_regression"]

PredKind = Literal["point", "proba", "quantiles", "interval"]
# "interval" = explicit (lo, hi) intervals, often the *output* of calibration


NDArrayF = npt.NDArray[np.float64]


# =========================
# Capability protocols
# =========================

@runtime_checkable
class SupportsPredict(Protocol):
    """Point prediction: y_hat shape (n_samples,)."""
    def predict(self, X: Any) -> NDArrayF: ...


@runtime_checkable
class SupportsPredictProba(Protocol):
    """Class probabilities: proba shape (n_samples, n_classes)."""
    def predict_proba(self, X: Any) -> NDArrayF: ...


@runtime_checkable
class SupportsPredictQuantiles(Protocol):
    """
    Quantile prediction: shape (n_samples, n_quantiles).
    If `quantiles` is None, use model's internal quantiles order.
    """
    def predict_quantiles(
        self,
        X: Any,
        quantiles: Sequence[float] | NDArrayF | None = None
    ) -> NDArrayF: ...


# =========================
# Predictions (discriminated union)
# =========================

@dataclass(frozen=True)
class PointPreds:
    kind: Literal["point"] = "point"
    y_hat: NDArrayF = np.empty(0)  # (n_samples,)


@dataclass(frozen=True)
class ProbPreds:
    kind: Literal["proba"] = "proba"
    proba: NDArrayF = np.empty((0, 0))          # (n_samples, n_classes)
    classes_: Sequence[Any] = ()                # Ordering of columns in `proba`


@dataclass(frozen=True)
class QuantilePreds:
    kind: Literal["quantiles"] = "quantiles"
    q: NDArrayF = np.empty((0, 0))              # (n_samples, n_quantiles)
    quantiles: Sequence[float] = ()             # same order as columns in `q`


@dataclass(frozen=True)
class IntervalPreds:
    kind: Literal["interval"] = "interval"
    lo: NDArrayF = np.empty(0)                  # (n_samples,)
    hi: NDArrayF = np.empty(0)                  # (n_samples,)
    target_coverage: float | None = None        # e.g., 0.80


PredictionBundle = PointPreds | ProbPreds | QuantilePreds | IntervalPreds


# Tiny type guards for ergonomic branching in evaluators/calibrators
def is_point(p: PredictionBundle) -> TypeGuard[PointPreds]: return p.kind == "point"
def is_proba(p: PredictionBundle) -> TypeGuard[ProbPreds]: return p.kind == "proba"
def is_quantiles(p: PredictionBundle) -> TypeGuard[QuantilePreds]: return p.kind == "quantiles"
def is_interval(p: PredictionBundle) -> TypeGuard[IntervalPreds]: return p.kind == "interval"


# =========================
# Evaluation result
# =========================

@runtime_checkable
class EvaluationResultProtocol(Protocol):
    @property
    def task_type(self) -> TaskType: ...
    @property
    def primary_metric_name(self) -> str: ...
    @property
    def primary_metric_value(self) -> float: ...
    @property
    def metrics(self) -> Mapping[str, Any]: ...
    @property
    def n_samples(self) -> int: ...
    # Optional: artifacts/curves (ROC pts, PR pts, residual histograms, etc.)
    @property
    def artifacts(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class EvaluationResult(EvaluationResultProtocol):
    task_type: TaskType
    primary_metric_name: str
    primary_metric_value: float
    metrics: Mapping[str, Any]
    n_samples: int
    artifacts: Mapping[str, Any] = None  # type: ignore[assignment]


# =========================
# Evaluator protocol
# =========================

@runtime_checkable
class ModelEvaluatorProtocol(Protocol):
    """
    Evaluators are task-/kind-specific, *stateless* objects that consume ground truth
    and a PredictionBundle, returning an EvaluationResult.
    """
    task_type: TaskType
    accepted_kinds: tuple[PredKind, ...]  # e.g. ("point",) or ("proba",) or ("quantiles","interval")

    def evaluate_predictions(
        self,
        y_true: NDArrayF,
        preds: PredictionBundle,
        *,
        sample_weight: NDArrayF | None = None,
        **kwargs: Any,
    ) -> EvaluationResult: ...


# =========================
# Calibrator protocol
# =========================

CalibrationKind = Literal[
    "probability",         # e.g., Platt/Isotonic for classification
    "quantile_interval",   # e.g., conformalizing (lo, hi)
    "regression_residual", # e.g., std calibration for Gaussian residuals
    "temperature",         # deep nets (logits scaling)
]

@runtime_checkable
class CalibratorProtocol(Protocol):
    """
    Calibrators *wrap* a fitted base estimator and return a new estimator with
    the same exposed prediction surface (or a superset), but calibrated.
    """
    calibration_kind: CalibrationKind
    task_type: TaskType
    # What this calibrator EXPECTS as input during fitting
    input_kind: tuple[PredKind, ...]
    # What the returned wrapper will EXPOSE as public prediction API
    output_kind: tuple[PredKind, ...]

    def fit(
        self,
        base_estimator: BaseEstimator,  # fitted
        X_cal: Any,
        y_cal: Any,
        *,
        extract: PredictionExtractor | None = None,
        **kwargs: Any,
    ) -> CalibratorProtocol: ...

    def wrap(self, base_estimator: BaseEstimator) -> BaseEstimator:
        """
        Return a sklearn-compatible estimator that performs calibrated predictions.

        - If output_kind contains "point" => wrapper must implement .predict(X)
        - If "proba"                      => wrapper must implement .predict_proba(X)
        - If "quantiles"                  => wrapper must implement .predict_quantiles(X, quantiles=None)
        - If "interval"                   => wrapper must implement .predict_interval(X, coverage=None) [optional]
        """
        ...

    def info(self) -> Mapping[str, Any]: ...


# =========================
# Task config (factory of defaults)
# =========================

@runtime_checkable
class TaskConfigProtocol(Protocol):
    """
    Narrow purpose: surface defaults & factories without importing concrete classes here.
    Implementations live elsewhere (e.g., classification_config.py).
    """
    @property
    def task_type(self) -> TaskType: ...
    @property
    def primary_metric_name(self) -> str: ...
    @property
    def default_quantiles(self) -> Sequence[float] | None: ...
    @property
    def prefers_calibration(self) -> bool: ...

    def make_default_evaluator(self) -> ModelEvaluatorProtocol: ...
    def make_default_calibrator(self) -> CalibratorProtocol | None: ...


# =========================
# Extractors (used by flows or calibrators)
# =========================

# A function that, given a fitted estimator and X, produces a typed PredictionBundle
PredictionExtractor = Callable[[BaseEstimator, Any], PredictionBundle]


def extract_point(est: BaseEstimator, X: Any) -> PointPreds:
    assert isinstance(est, SupportsPredict), (
        "Estimator does not implement predict(X)."
    )
    y_hat = _to_1d(np.asarray(est.predict(X)))
    return PointPreds(y_hat=y_hat)


def extract_proba(est: BaseEstimator, X: Any, classes: Sequence[Any] | None = None) -> ProbPreds:
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
    assert isinstance(est, SupportsPredictQuantiles), (
        "Estimator does not implement predict_quantiles(X, quantiles=None)."
    )
    q = np.asarray(est.predict_quantiles(X, quantiles=quantiles), dtype=float)
    q_levels = tuple(quantiles) if quantiles is not None else tuple(range(q.shape[1]))
    return QuantilePreds(q=q, quantiles=q_levels)


# =========================
# Optional (lightweight) interval surface
# =========================

@runtime_checkable
class SupportsPredictInterval(Protocol):
    """
    Optional surface that calibrated wrappers may expose.
    Returns (lo, hi) interval achieving requested coverage if provided;
    otherwise uses trained default coverage (if any).
    """
    def predict_interval(
        self, X: Any, coverage: float | None = None
    ) -> tuple[NDArrayF, NDArrayF]: ...


# =========================
# Sklearn wrapper helper (mixin)
# =========================

class GetSetParamsMixin:
    """
    A tiny helper mixin so your wrappers remain sklearn-compatible.
    Use it in your calibrated wrappers or custom estimators.
    """
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        # expose all public attributes (shallow); override for more control
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def set_params(self, **params: Any) -> "GetSetParamsMixin":
        for k, v in params.items():
            setattr(self, k, v)
        return self


# =========================
# Utilities
# =========================

def _to_1d(a: NDArrayF) -> NDArrayF:
    """Ravel to (n,), copying as float64."""
    return np.asarray(a, dtype=float).reshape(-1)

def ensure_monotone_quantiles(q: NDArrayF, axis: int = 1) -> bool:
    """Quick monotonicity check for quantiles along axis (no penalization logic here)."""
    diffs = np.diff(q, axis=axis)
    return bool(np.all(diffs >= -1e-12))  # tiny tolerance

def validate_prediction_bundle_size(preds: PredictionBundle, n: int) -> None:
    """Raise if the bundle size doesn't match n samples."""
    if is_point(preds):
        if preds.y_hat.shape[0] != n:
            raise ValueError("PointPreds length mismatch.")
    elif is_proba(preds):
        if preds.proba.shape[0] != n:
            raise ValueError("ProbPreds length mismatch.")
    elif is_quantiles(preds):
        if preds.q.shape[0] != n:
            raise ValueError("QuantilePreds length mismatch.")
    elif is_interval(preds):
        if preds.lo.shape[0] != n or preds.hi.shape[0] != n:
            raise ValueError("IntervalPreds length mismatch.")
    else:
        raise TypeError("Unknown PredictionBundle kind.")
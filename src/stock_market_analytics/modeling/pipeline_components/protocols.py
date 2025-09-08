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
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
    TypeGuard,
)

import numpy as np
import numpy.typing as npt
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
    y_hat: NDArrayF = None  # (n_samples,)
    
    def __post_init__(self):
        if self.y_hat is None:
            object.__setattr__(self, 'y_hat', np.empty(0, dtype=float))


@dataclass(frozen=True)
class ProbPreds:
    kind: Literal["proba"] = "proba"
    proba: NDArrayF = None          # (n_samples, n_classes)
    classes_: Sequence[Any] = None  # Ordering of columns in `proba`
    
    def __post_init__(self):
        if self.proba is None:
            object.__setattr__(self, 'proba', np.empty((0, 0), dtype=float))
        if self.classes_ is None:
            object.__setattr__(self, 'classes_', ())


@dataclass(frozen=True)
class QuantilePreds:
    kind: Literal["quantiles"] = "quantiles"
    q: NDArrayF = None              # (n_samples, n_quantiles)
    quantiles: Sequence[float] = None  # same order as columns in `q`
    
    def __post_init__(self):
        if self.q is None:
            object.__setattr__(self, 'q', np.empty((0, 0), dtype=float))
        if self.quantiles is None:
            object.__setattr__(self, 'quantiles', ())


@dataclass(frozen=True)
class IntervalPreds:
    kind: Literal["interval"] = "interval"
    lo: NDArrayF = None                  # (n_samples,)
    hi: NDArrayF = None                  # (n_samples,)
    target_coverage: float | None = None # e.g., 0.80
    
    def __post_init__(self):
        if self.lo is None:
            object.__setattr__(self, 'lo', np.empty(0, dtype=float))
        if self.hi is None:
            object.__setattr__(self, 'hi', np.empty(0, dtype=float))


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
    task_type: TaskType
    primary_metric_name: str
    primary_metric_value: float
    metrics: Mapping[str, Any]
    n_samples: int
    artifacts: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationResult(EvaluationResultProtocol):
    task_type: TaskType
    primary_metric_name: str
    primary_metric_value: float
    metrics: Mapping[str, Any]
    n_samples: int
    artifacts: Mapping[str, Any] | None = None
    
    def __post_init__(self):
        if self.artifacts is None:
            object.__setattr__(self, 'artifacts', {})


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
# Data splitting protocol
# =========================

@dataclass(frozen=True)
class DataSplit:
    """Container for a data split with metadata."""
    name: str                                    # e.g., "train", "val", "test", "fold_1", "outer_train"
    indices: NDArrayF                            # Row indices for this split
    metadata: Mapping[str, Any] | None = None    # Optional metadata (date ranges, etc.)


@runtime_checkable
class DataSplitterProtocol(Protocol):
    """
    Protocol for data splitting strategies.
    
    Supports various splitting scenarios:
    - Simple train/val/test
    - K-fold cross-validation  
    - Time series splits (expanding window, sliding window)
    - Nested cross-validation
    - Custom splits with arbitrary fold names
    """
    
    def split_data(
        self, 
        data: Any,
        target: Any | None = None,
        groups: Any | None = None,
        **kwargs: Any
    ) -> list[DataSplit]:
        """
        Split data into multiple named splits.
        
        Args:
            data: Input data (DataFrame, array, etc.)
            target: Optional target variable for stratified splits
            groups: Optional group labels for group-based splits
            **kwargs: Additional splitter-specific parameters
            
        Returns:
            List of DataSplit objects, each containing:
            - name: Split identifier ("train", "val", "test", "fold_1", etc.)
            - indices: Row indices for this split
            - metadata: Optional split-specific information
        """
        ...
    
    @property
    def split_names(self) -> list[str]:
        """Return the expected split names this splitter produces."""
        ...


# =========================
# Model factory protocol  
# =========================

@runtime_checkable
class ModelFactoryProtocol(Protocol):
    """Protocol for creating different model types with consistent interfaces."""
    
    def create_model(
        self, 
        model_type: str, 
        **kwargs: Any
    ) -> BaseEstimator:
        """
        Create a model of the specified type.
        
        Args:
            model_type: Model identifier (e.g., "catboost", "linear", "historical")
            **kwargs: Model-specific parameters
            
        Returns:
            Fitted or unfitted BaseEstimator
        """
        ...
    
    def get_available_models(self) -> list[str]:
        """Return list of available model types."""
        ...
    
    def get_model_info(self, model_type: str) -> Mapping[str, Any]:
        """Get metadata about a specific model type."""
        ...


# =========================
# Utilities
# =========================


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
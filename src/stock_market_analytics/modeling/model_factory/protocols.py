"""
Protocols for a model factory (pandas/NumPy only).

Conventions
-----------
- Frames/series: pandas only (no polars).
- Arrays: NumPy float arrays unless stated otherwise.
- Shapes:
    * Regression predictions: (n_samples,)
    * Multiclass probabilities: (n_samples, n_classes)
    * Quantiles: (n_samples, n_quantiles)
    * Index splits: integer index arrays
- Calibration maps model outputs -> calibrated outputs (X is optional context).

These are structural types (Protocols) to decouple components while preserving type safety.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, Mapping, Sequence, Iterator

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# ----- Common type aliases ----------------------------------------------------

Array = NDArray[np.float64]          # numeric arrays (preds, probs, quantiles)
IndexArray = NDArray[np.intp]        # index arrays for CV splitters
Frame = pd.DataFrame
Series = pd.Series
Metrics = Mapping[str, float]


# ----- Calibration ------------------------------------------------------------

@runtime_checkable
class BaseCalibrator(Protocol):
    """Learns a mapping from model outputs to calibrated outputs.

    Notes
    -----
    - `y_pred_cal` are the base model's outputs used to fit the calibrator:
        * Binary classification: scores/logits/probs shape (n,)
        * Multiclass: scores/logits/probs shape (n, C)
        * Quantiles: predicted quantiles shape (n, Q)
    - `X_cal` is optional, for context-aware calibration if needed.
    """

    def fit(
        self,
        y_pred_cal: Array,
        y_true_cal: Array,
        X_cal: Frame | None = None,
        **kwargs: Any,
    ) -> BaseCalibrator:
        ...

    def transform(self, y_pred: Array, **kwargs: Any) -> Array:
        """Map raw model outputs to calibrated outputs."""
        ...

    def fit_transform(
        self,
        y_pred_cal: Array,
        y_true_cal: Array,
        X_cal: Frame | None = None,
        **kwargs: Any,
    ) -> Array:
        ...


@runtime_checkable
class ProbabilityCalibrator(BaseCalibrator, Protocol):
    """Calibrate scores/logits/probabilities to calibrated probabilities.

    Expected output shapes
    ----------------------
    - Binary: (n,) or (n, 1)
    - Multiclass: (n, C) with rows summing to ~1.0
    """

    def transform(self, y_score: Array, **kwargs: Any) -> Array:
        ...


@runtime_checkable
class QuantileCalibrator(BaseCalibrator, Protocol):
    """Calibrate multi-quantile predictions (n_samples, n_quantiles)."""

    def transform(self, y_pred_quantiles: Array, **kwargs: Any) -> Array:
        ...

# ----- Data splitting -------------------------------------------------------------#

@runtime_checkable
class DataSplitter(Protocol):
    """Protocol for data splitting strategies."""

    def split(self, df: pd.DataFrame) -> Iterator[Frame]:
        """Split the DataFrame into training and testing sets."""
        ...

# ----- Post-processing / business rules --------------------------------------

@runtime_checkable
class PostProcessor(Protocol):
    """Apply and validate business rules on predictions."""

    def apply_rules(
        self,
        predictions: Array,
        context: dict[str, Any] | None = None,
    ) -> Array:
        ...

    def validate_predictions(self, predictions: Array) -> bool:
        """Return True if predictions satisfy business constraints."""
        ...

    # Optional but useful for debugging/monitoring:
    def report_violations(self, predictions: Array) -> dict[str, Any]:  # pragma: no cover
        """Return a structured report of constraint violations (empty dict if none)."""
        ...


# ----- Evaluation -------------------------------------------------------------

@runtime_checkable
class ModelEvaluator(Protocol):
    """Standardized evaluation interface."""

    def evaluate(self, y_true: Array, y_pred: Array) -> dict[str, float]:
        ...

    def get_metric_names(self) -> list[str]:
        ...


@runtime_checkable
class QuantileEvaluator(ModelEvaluator, Protocol):
    """Evaluation for quantile/interval predictions."""

    def evaluate_quantiles(
        self,
        y_true: Array,
        y_pred_quantiles: Array,
        quantiles: Sequence[float],
    ) -> dict[str, float]:
        ...

    def evaluate_intervals(
        self,
        y_true: Array,
        y_lower: Array,
        y_upper: Array,
        alpha: float = 0.1,
    ) -> dict[str, float]:
        ...


# ----- Estimators -------------------------------------------------------------

@runtime_checkable
class SklearnCompatibleEstimator(Protocol):
    """Sklearn-compatible estimator surface (cloneable & tunable)."""

    def fit(self, X: Frame | Array, y: Series | Array, **kwargs: Any) -> SklearnCompatibleEstimator:
        ...

    def predict(self, X: Frame | Array) -> Array:
        ...

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        ...

    # Required for sklearn cloning/grid-search compatibility
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        ...

    def set_params(self, **params: Any) -> SklearnCompatibleEstimator:
        ...


@runtime_checkable
class SupportsFeatureImportances(Protocol):
    """Optional mixin for models exposing feature_importances_."""

    @property
    def feature_importances_(self) -> Array:
        ...


@runtime_checkable
class SupportsPredictProba(Protocol):
    """Optional mixin for classifiers exposing predict_proba."""

    def predict_proba(self, X: Frame | Array) -> Array:
        ...


@runtime_checkable
class QuantileEstimator(SklearnCompatibleEstimator, Protocol):
    """Estimators that predict multiple quantiles per sample."""

    quantiles: Sequence[float]

    def predict(self, X: Frame | Array) -> Array:
        """Return shape (n_samples, n_quantiles)."""
        ...

    def transform(self, X: Frame | Array) -> Array:
        """Alias for predict to enable transformer usage in pipelines."""
        ...


# ----- Pipeline ---------------------------------------------------------------

@runtime_checkable
class ModelingPipeline(Protocol):
    """End-to-end pipeline interface decoupled from concrete components."""

    def fit(self, X: Frame, y: Series) -> ModelingPipeline:
        ...

    def predict(self, X: Frame) -> Array:
        ...

    def evaluate(self, X: Frame, y: Series, evaluator: ModelEvaluator) -> dict[str, float]:
        ...

    def get_components(self) -> dict[str, Any]:
        """Return named subcomponents (e.g., {'preprocessor': ..., 'estimator': ..., 'calibrator': ...})."""
        ...


__all__ = [
    # aliases
    "Array", "IndexArray", "Frame", "Series", "Metrics",
    # calibrators
    "BaseCalibrator", "ProbabilityCalibrator", "QuantileCalibrator",
    # post-processing
    "PostProcessor",
    # evaluation
    "ModelEvaluator", "QuantileEvaluator",
    # estimators
    "SklearnCompatibleEstimator", "SupportsFeatureImportances", "SupportsPredictProba", "QuantileEstimator",
    # pipeline
    "ModelingPipeline",
]
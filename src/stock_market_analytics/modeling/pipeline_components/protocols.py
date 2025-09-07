"""
Generalized protocols for pipeline components to enable extensibility across different ML tasks.

This module defines abstract interfaces that allow the same modeling workflow to work
with different types of models (regression, classification, etc.) and post-processing
techniques (calibration, uncertainty quantification, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@runtime_checkable
class EvaluationResult(Protocol):
    """
    Standardized result format for all model evaluation operations.
    
    This protocol ensures that all evaluators return a consistent format that
    can be easily consumed by modeling_steps.py and other components.
    """
    
    @property
    def primary_metric(self) -> float:
        """
        The primary metric for this evaluation task.
        
        For regression: typically MSE, MAE, or pinball loss
        For classification: typically accuracy, F1, or log-loss
        For quantile regression: typically pinball loss
        """
        ...
    
    @property
    def metrics(self) -> dict[str, Any]:
        """
        Complete dictionary of all evaluation metrics.
        
        Should include:
        - All computed metrics (e.g., accuracy, precision, coverage, etc.)
        - Task-specific information (e.g., quantiles for quantile regression)
        - Any debugging/analysis information
        """
        ...
    
    @property
    def task_type(self) -> str:
        """The type of ML task (e.g., 'quantile_regression', 'classification', 'regression')."""
        ...


class BaseEvaluationResult:
    """
    Base implementation of EvaluationResult protocol.
    
    This provides a concrete implementation that all evaluators can use or extend.
    """
    
    def __init__(
        self, 
        primary_metric: float,
        metrics: dict[str, Any],
        task_type: str
    ):
        self._primary_metric = primary_metric
        self._metrics = metrics
        self._task_type = task_type
    
    @property
    def primary_metric(self) -> float:
        return self._primary_metric
    
    @property 
    def metrics(self) -> dict[str, Any]:
        return self._metrics
    
    @property
    def task_type(self) -> str:
        return self._task_type
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"primary_metric={self.primary_metric:.4f}, "
                f"task_type='{self.task_type}')")


@runtime_checkable  
class ModelEvaluatorProtocol(Protocol):
    """
    Protocol for model evaluation components.
    
    This protocol allows different evaluators (quantile regression, classification, 
    standard regression, etc.) to be used interchangeably in the modeling workflow.
    """
    
    def evaluate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        **kwargs: Any
    ) -> EvaluationResult:
        """
        Evaluate model performance on given data.
        
        Args:
            pipeline: Fitted sklearn pipeline
            X: Features
            y: Target values
            **kwargs: Additional evaluation parameters (sample_weight, etc.)
            
        Returns:
            EvaluationResult containing primary metric and detailed metrics
        """
        ...


@runtime_checkable
class CalibratorProtocol(Protocol):
    """
    Protocol for model calibration components.
    
    This protocol allows different calibration techniques to be used interchangeably:
    - Conformal prediction for quantile regression
    - Platt scaling for classification  
    - Temperature scaling for deep learning
    - Isotonic regression for general calibration
    """
    
    def fit_calibrator(
        self,
        base_pipeline: Pipeline,
        X_cal: pd.DataFrame | np.ndarray, 
        y_cal: pd.DataFrame | np.ndarray,
        **kwargs: Any
    ) -> 'CalibratorProtocol':
        """
        Fit the calibrator using calibration data.
        
        Args:
            base_pipeline: Fitted base model pipeline
            X_cal: Calibration features
            y_cal: Calibration targets
            **kwargs: Additional calibrator-specific parameters
            
        Returns:
            Self for method chaining
        """
        ...
    
    def create_calibrated_pipeline(
        self,
        base_pipeline: Pipeline
    ) -> Pipeline:
        """
        Create a new pipeline with calibration as the final step.
        
        Args:
            base_pipeline: The base fitted pipeline
            
        Returns:
            New pipeline with calibrator added as final step
        """
        ...
    
    def get_calibration_info(self) -> dict[str, Any]:
        """
        Get information about the fitted calibrator.
        
        Returns:
            Dictionary with calibrator-specific information
        """
        ...


class BaseCalibrator(ABC):
    """
    Abstract base class for calibrator implementations.
    
    This provides common functionality and ensures all calibrators
    follow the same interface patterns.
    """
    
    def __init__(self, **kwargs: Any):
        self.is_fitted_ = False
        self._calibration_info = {}
    
    @abstractmethod
    def _fit_calibrator_impl(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        **kwargs: Any
    ) -> None:
        """
        Task-specific calibrator fitting logic.
        
        Args:
            predictions: Model predictions on calibration set
            targets: True targets for calibration set
            **kwargs: Additional parameters
        """
        pass
    
    @abstractmethod
    def _create_calibrator_component(self) -> Any:
        """
        Create the sklearn-compatible calibrator component.
        
        Returns:
            Sklearn transformer/predictor that performs calibration
        """
        pass
    
    def fit_calibrator(
        self,
        base_pipeline: Pipeline,
        X_cal: pd.DataFrame | np.ndarray,
        y_cal: pd.DataFrame | np.ndarray,
        **kwargs: Any
    ) -> 'BaseCalibrator':
        """Fit the calibrator using calibration data."""
        # Get predictions from base pipeline
        predictions = base_pipeline.predict(X_cal)
        
        # Convert targets to numpy
        targets = (
            y_cal.to_numpy().ravel() if hasattr(y_cal, 'values') 
            else np.asarray(y_cal).ravel()
        )
        
        # Fit calibrator
        self._fit_calibrator_impl(predictions, targets, **kwargs)
        self.is_fitted_ = True
        
        return self
    
    def create_calibrated_pipeline(self, base_pipeline: Pipeline) -> Pipeline:
        """Create calibrated pipeline."""
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted before creating pipeline")
            
        calibrator_component = self._create_calibrator_component()
        
        # Add calibrator as final step
        new_steps = base_pipeline.steps + [("calibrator", calibrator_component)]
        return Pipeline(new_steps)
    
    def get_calibration_info(self) -> dict[str, Any]:
        """Get calibration information."""
        return self._calibration_info.copy()


@runtime_checkable
class TaskConfigProtocol(Protocol):
    """
    Protocol for task-specific configuration.
    
    This allows different ML tasks to specify their requirements
    (metrics, calibration methods, etc.) in a standardized way.
    """
    
    @property
    def task_type(self) -> str:
        """The type of ML task (e.g., 'quantile_regression', 'classification')."""
        ...
    
    @property  
    def primary_metric(self) -> str:
        """Name of the primary metric for this task."""
        ...
    
    @property
    def requires_calibration(self) -> bool:
        """Whether this task typically requires calibration."""
        ...
    
    def get_default_evaluator(self) -> ModelEvaluatorProtocol:
        """Get the default evaluator for this task."""
        ...
    
    def get_default_calibrator(self) -> CalibratorProtocol | None:
        """Get the default calibrator for this task (if any)."""
        ...
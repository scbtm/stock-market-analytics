"""
Pipeline components for stock market analytics modeling.

This module provides protocol-compliant components for flexible ML experimentation:
- Protocols for model interfaces and evaluation
- Predictors with quantile regression capabilities
- Evaluators for model performance assessment
- Calibrators for uncertainty quantification
- Data splitters and model factories for experimentation
"""

# Core protocols
from .protocols import (
    # Capability protocols
    SupportsPredict,
    SupportsPredictProba,
    SupportsPredictQuantiles,
    SupportsPredictInterval,
    
    # Prediction bundles
    PredictionBundle,
    PointPreds,
    ProbPreds,
    QuantilePreds,
    IntervalPreds,
    
    # Evaluation
    EvaluationResult,
    ModelEvaluatorProtocol,
    
    # Calibration
    CalibratorProtocol,
    
    # Data and model management
    DataSplitterProtocol,
    DataSplit,
    ModelFactoryProtocol,
    TaskConfigProtocol,
    
    # Utilities
    PredictionExtractor,
    extract_point,
    extract_proba,
    extract_quantiles,
    
    # Type guards
    is_point,
    is_proba,
    is_quantiles,
    is_interval,
)

# Concrete implementations
from .predictors import CatBoostMultiQuantileModel
from .naive_baselines import HistoricalQuantileBaseline
from .evaluators import (
    QuantileRegressionEvaluator,
    EvaluationReport,
)
from .calibrators import (
    QuantileIntervalCalibrator,
    CalibratedQuantileWrapper,
)
from .factories import (
    TimeSeriesDataSplitter,
    QuantileRegressionModelFactory,
    QuantileRegressionTaskConfig,
)

__all__ = [
    # Protocols
    "SupportsPredict",
    "SupportsPredictProba", 
    "SupportsPredictQuantiles",
    "SupportsPredictInterval",
    "PredictionBundle",
    "PointPreds",
    "ProbPreds", 
    "QuantilePreds",
    "IntervalPreds",
    "EvaluationResult",
    "ModelEvaluatorProtocol",
    "CalibratorProtocol",
    "DataSplitterProtocol",
    "DataSplit",
    "ModelFactoryProtocol",
    "TaskConfigProtocol",
    "PredictionExtractor",
    "extract_point",
    "extract_proba",
    "extract_quantiles",
    "is_point",
    "is_proba",
    "is_quantiles",
    "is_interval",
    
    # Predictors
    "CatBoostMultiQuantileModel",
    "HistoricalQuantileBaseline",
    
    # Evaluators
    "QuantileRegressionEvaluator", 
    "EvaluationReport",
    
    # Calibrators
    "QuantileIntervalCalibrator",
    "CalibratedQuantileWrapper",
    
    # Factories and splitters
    "TimeSeriesDataSplitter",
    "QuantileRegressionModelFactory",
    "QuantileRegressionTaskConfig",
]
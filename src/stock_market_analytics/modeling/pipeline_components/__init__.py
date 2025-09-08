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
    
    # Type guards
    is_point,
    is_proba,
    is_quantiles,
    is_interval,
)

# Utilities
from .protocol_functions import (
    PredictionExtractor,
    extract_point,
    extract_proba,
    extract_quantiles,
)

# Concrete implementations  
from .prediction import (
    CatBoostMultiQuantileModel, 
    HistoricalQuantileBaseline,
    QuantileRegressionModelFactory,
)
from .evaluation import (
    QuantileRegressionEvaluator,
)
from .calibration import (
    QuantileIntervalCalibrator,
    CalibratedQuantileWrapper,
)
from .data_management import (
    TimeSeriesDataSplitter,
)
from .workflows import (
    QuantileRegressionTaskConfig,
    create_preprocessing_pipeline,
    get_pipeline,
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
    
    # Calibrators
    "QuantileIntervalCalibrator",
    "CalibratedQuantileWrapper",
    
    # Data management
    "TimeSeriesDataSplitter",
    
    # Workflows and factories  
    "QuantileRegressionModelFactory",
    "QuantileRegressionTaskConfig",
    "create_preprocessing_pipeline",
    "get_pipeline",
]
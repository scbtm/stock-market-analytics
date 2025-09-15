"""
Stock Market Analytics - Modeling Module

This module contains all machine learning and modeling components for the
stock market analytics project. It exports the public interfaces from
the model_factory submodules following the established architecture pattern.
"""

# Export estimators from model_factory
# Export postprocessors from model_factory
# from .model_factory.data_management.postprocessors import (
#     ReturnConstraintProcessor,
#     QuantileConsistencyProcessor,
#     OutlierClippingProcessor,
#     MarketRegimeProcessor,
#     CompositePredictionProcessor,
# )
# Export calibrators from model_factory
from .model_factory.calibration.calibrators import (
    ConformalizedQuantileCalibrator,
    QuantileConformalCalibrator,
)
from .model_factory.estimation.estimators import (
    CatBoostMultiQuantileModel,
    # CatBoostQuantileRegressor,
    # LightGBMQuantileRegressor,
    # RandomForestQuantileRegressor,
    # LinearQuantileRegressor,
    # BaselineEstimator,
)

# Export evaluators from model_factory
from .model_factory.evaluation.evaluators import (
    QuantileRegressionEvaluator,
    # RegressionEvaluator,
    # FinancialRegressionEvaluator,
    # QuantileRegressionEvaluator,
    # ClassificationEvaluator,
    # CompositeEvaluator,
    # BacktestEvaluator,
)

# Export protocols for type safety
from .model_factory.protocols import (
    BaseCalibrator,
    DataSplitter,
    ModelEvaluator,
    ModelingPipeline,
    PostProcessor,
    ProbabilityCalibrator,
    QuantileCalibrator,
    QuantileEstimator,
    QuantileEvaluator,
    SklearnCompatibleEstimator,
)

__all__ = [
    # Estimators
    "CatBoostMultiQuantileModel",
    # Evaluators
    "QuantileRegressionEvaluator",
    # Calibrators
    "QuantileConformalCalibrator",
    "ConformalizedQuantileCalibrator",
    # Protocols
    "BaseCalibrator",
    "QuantileCalibrator",
    "ProbabilityCalibrator",
    "DataSplitter",
    "PostProcessor",
    "ModelEvaluator",
    "QuantileEvaluator",
    "SklearnCompatibleEstimator",
    "QuantileEstimator",
    "ModelingPipeline",
]

"""
Stock Market Analytics - Modeling Module

This module contains all machine learning and modeling components for the
stock market analytics project. It exports the public interfaces from
the model_factory submodules following the established architecture pattern.
"""

# Export estimators from model_factory
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

# Export splitters from model_factory
from .model_factory.data_management.preprocessors import (
    PurgedTimeSeriesSplit,
    PanelHorizonSplitter,
    # TimeSeriesSplitter,
    # WalkForwardSplitter,
    # StratifiedTimeSeriesSplitter,
    # GroupedSplitter,
)

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
    QuantileConformalCalibrator,
    # AdaptiveConformalCalibrator,
    # PlattScalingCalibrator,
    # IsotonicRegressionCalibrator,
)

# Export protocols for type safety
from .model_factory.protocols import (
    Calibrator,
    QuantileCalibrator,
    ProbabilityCalibrator,
    DataSplitter,
    PostProcessor,
    ModelEvaluator,
    QuantileEvaluator,
    SklearnCompatibleEstimator,
    QuantileEstimator,
    ModelingPipeline,
)

__all__ = [
    # Estimators
    "CatBoostMultiQuantileModel",
    # "CatBoostQuantileRegressor",
    # "LightGBMQuantileRegressor",
    # "RandomForestQuantileRegressor",
    # "LinearQuantileRegressor",
    # "BaselineEstimator",
    # Evaluators
    "QuantileRegressionEvaluator",
    # "RegressionEvaluator",
    # "FinancialRegressionEvaluator",
    # "QuantileRegressionEvaluator",
    # "ClassificationEvaluator",
    # "CompositeEvaluator",
    # "BacktestEvaluator",
    # Data Management
    "PurgedTimeSeriesSplit",
    "PanelHorizonSplitter",
    # "TimeSeriesSplitter",
    # "WalkForwardSplitter",
    # "StratifiedTimeSeriesSplitter",
    # "GroupedSplitter",
    # "ReturnConstraintProcessor",
    # "QuantileConsistencyProcessor",
    # "OutlierClippingProcessor",
    # "MarketRegimeProcessor",
    # "CompositePredictionProcessor",
    # Calibrators
    "QuantileConformalCalibrator",
    # "AdaptiveConformalCalibrator",
    # "PlattScalingCalibrator",
    # "IsotonicRegressionCalibrator",
    # Protocols
    "Calibrator",
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

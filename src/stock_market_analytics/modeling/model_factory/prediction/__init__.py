"""
Prediction components for stock market analytics modeling.

Clean, focused implementations without unnecessary factory patterns.
"""

from .predictors import CatBoostMultiQuantileModel
from .baseline_predictors import HistoricalQuantileBaseline
from .prediction_functions import (
    create_catboost_pool,
    predict_quantiles_catboost,
    detect_categorical_features,
)

__all__ = [
    # Models
    "CatBoostMultiQuantileModel",
    "HistoricalQuantileBaseline",
    
    # Utility functions
    "create_catboost_pool",
    "predict_quantiles_catboost", 
    "detect_categorical_features",
]
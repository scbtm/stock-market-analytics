"""
Model factories for prediction components.

This module provides factories for creating different types of predictors
with consistent interfaces for experimentation.
"""

from typing import Any, Mapping
from sklearn.base import BaseEstimator

from stock_market_analytics.config import config
from stock_market_analytics.modeling.pipeline_components.protocols import (
    ModelFactoryProtocol,
)
from .predictors import CatBoostMultiQuantileModel
from .baseline_predictors import HistoricalQuantileBaseline


class QuantileRegressionModelFactory(ModelFactoryProtocol):
    """
    Factory for creating quantile regression models.
    
    Supports various model types with consistent interfaces for experimentation.
    """
    
    def __init__(self, quantiles: list[float] | None = None):
        """
        Initialize model factory.
        
        Args:
            quantiles: Default quantiles for all models
        """
        self.quantiles = quantiles or config.modeling.quantiles
    
    def get_available_models(self) -> list[str]:
        """Return list of available model types."""
        return ["catboost", "historical"]
    
    def create_model(
        self, 
        model_type: str, 
        **kwargs: Any
    ) -> BaseEstimator:
        """
        Create a model of the specified type.
        
        Args:
            model_type: Model identifier ("catboost", "historical", etc.)
            **kwargs: Model-specific parameters
            
        Returns:
            Unfitted BaseEstimator implementing SupportsPredictQuantiles
        """
        # Merge default quantiles with kwargs
        model_params = {"quantiles": self.quantiles}
        model_params.update(kwargs)
        
        if model_type == "catboost":
            # Add CatBoost-specific parameters from config
            cb_params = config.modeling.cb_model_params.copy()
            cb_params.update(model_params)
            return CatBoostMultiQuantileModel(**cb_params)
        
        elif model_type == "historical":
            return HistoricalQuantileBaseline(**model_params)
        
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Available options: {self.get_available_models()}"
            )
    
    def get_model_info(self, model_type: str) -> Mapping[str, Any]:
        """Get metadata about a specific model type."""
        if model_type == "catboost":
            return {
                "name": "CatBoost Multi-Quantile Regressor",
                "type": "gradient_boosting",
                "supports_early_stopping": True,
                "supports_categorical": True,
                "default_params": config.modeling.cb_model_params,
            }
        
        elif model_type == "historical":
            return {
                "name": "Historical Quantile Baseline",
                "type": "naive_baseline",
                "supports_early_stopping": False,
                "supports_categorical": False,
                "default_params": {},
            }
        
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Available options: {self.get_available_models()}"
            )
"""
Complete pipeline assembly for stock market analytics modeling.

This module provides high-level functions for creating complete ML pipelines
with preprocessing and models.
"""

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_market_analytics.config import config
from stock_market_analytics.modeling.model_factory.prediction import (
    CatBoostMultiQuantileModel,
    HistoricalQuantileBaseline,
)


def create_preprocessing_pipeline():
    """Create a standardized preprocessing pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=1)),  # Retain 95% variance
    ])


def get_pipeline(model_type: str = "catboost", **model_params) -> Pipeline:
    """
    Create a complete ML pipeline with preprocessing and model.
    
    Args:
        model_type: Type of model to use (e.g., "catboost", "historical")
        **model_params: Additional parameters to pass to model creation
        
    Returns:
        Complete sklearn Pipeline
    """
    # Create preprocessing pipeline
    preprocessing = create_preprocessing_pipeline()
    
    # Create model directly - no factory needed for 2 models
    if model_type == "catboost":
        # Merge default quantiles and config parameters
        cb_params = config.modeling.cb_model_params.copy()
        cb_params.update(model_params)
        if "quantiles" not in cb_params:
            cb_params["quantiles"] = config.modeling.quantiles
        model = CatBoostMultiQuantileModel(**cb_params)
    
    elif model_type == "historical":
        hist_params = model_params.copy()
        if "quantiles" not in hist_params:
            hist_params["quantiles"] = config.modeling.quantiles
        model = HistoricalQuantileBaseline(**hist_params)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Available: ['catboost', 'historical']")
    
    # Create complete pipeline
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])





"""
Shared prediction utilities for models.

This module contains reusable prediction functions that can be shared
across different model implementations.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from typing import Any


def detect_categorical_features(X: Any) -> np.ndarray | None:
    """Detect categorical features from DataFrame."""
    if hasattr(X, "dtypes"):
        return np.where((X.dtypes == "category") | (X.dtypes == "object"))[0]
    return None


def create_catboost_pool(X: Any, y: Any = None) -> Pool:
    """Create CatBoost Pool with automatic categorical feature detection."""
    cat_features = detect_categorical_features(X)
    return Pool(X, y, cat_features=cat_features)


def predict_quantiles_catboost(model: CatBoostRegressor, X: Any) -> np.ndarray:
    """
    Make quantile predictions using a CatBoost model.
    
    Args:
        model: Trained CatBoost regressor configured for quantile regression
        X: Input features (DataFrame, array, or Pool)
        
    Returns:
        Array of shape (n_samples, n_quantiles) with monotonic quantile predictions
    """
    if isinstance(X, Pool):
        pool = X
    else:
        pool = create_catboost_pool(X)

    predictions = model.predict(pool)
    predictions = np.asarray(predictions)
    
    # Ensure proper shape
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    
    # Enforce non-crossing quantiles (simple sort)
    predictions.sort(axis=1)
    return predictions
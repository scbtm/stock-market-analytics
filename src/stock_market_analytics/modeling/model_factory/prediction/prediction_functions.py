"""
Shared prediction utilities for models.

This module contains reusable prediction functions that can be shared
across different model implementations.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from typing import Any


def validate_quantiles(quantiles: list[float]) -> None:
    """Basic validation for quantile values."""
    if not all(0 <= q <= 1 for q in quantiles):
        raise ValueError("Quantiles must be between 0 and 1")
    if quantiles != sorted(quantiles):
        raise ValueError("Quantiles must be sorted")


def detect_categorical_features(X: Any) -> np.ndarray | None:
    """Detect categorical features from DataFrame."""
    if hasattr(X, "dtypes"):
        return np.where((X.dtypes == "category") | (X.dtypes == "object"))[0]
    return None


def create_catboost_pool(X: Any, y: Any = None) -> Pool:
    """Create CatBoost Pool with automatic categorical feature detection."""
    cat_features = detect_categorical_features(X)
    return Pool(X, y, cat_features=cat_features)


# predict_quantiles_catboost function removed - it was unnecessary abstraction
# The logic is now inline in CatBoostMultiQuantileModel.predict_quantiles()
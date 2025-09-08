"""
Core prediction functions for ML models.

This module contains model-specific prediction utilities, particularly
for CatBoost quantile regression models.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


def predict_quantiles(model: CatBoostRegressor, X: pd.DataFrame | Pool) -> np.ndarray:
    """
    Make quantile predictions using a CatBoost model.
    
    MultiQuantile returns (n_samples, n_quantiles).
    Automatically handles categorical features and enforces monotonic quantiles.
    
    Args:
        model: Trained CatBoost regressor configured for quantile regression
        X: Input features as DataFrame or CatBoost Pool
        
    Returns:
        Array of shape (n_samples, n_quantiles) with monotonic quantile predictions
    """
    is_df = isinstance(X, pd.DataFrame)

    if is_df:
        cat_idx = np.where((X.dtypes == "category") | (X.dtypes == "object"))[0]
        pool = Pool(X, cat_features=cat_idx)
    else:
        pool = X

    qhat = model.predict(pool)
    qhat = np.asarray(qhat)
    # Enforce non-crossing (cheap rearrangement)
    qhat.sort(axis=1)
    return qhat
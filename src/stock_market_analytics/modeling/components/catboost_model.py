"""
Clean CatBoost multi-quantile model wrapper.

This module provides a clean interface around CatBoost for multi-quantile regression
with proper error handling, validation, and reusability for inference.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

logger = logging.getLogger(__name__)


class CatBoostMultiQuantileModel:
    """
    Clean wrapper around CatBoost for multi-quantile regression.
    
    This class encapsulates all CatBoost-specific logic and provides a clean interface
    that can be reused for both training and inference.
    """
    
    def __init__(
        self, 
        quantiles: List[float],
        random_state: int = 42,
        **catboost_params: Any
    ):
        """
        Initialize CatBoost multi-quantile model.
        
        Args:
            quantiles: List of quantiles to predict (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])
            random_state: Random seed for reproducibility
            **catboost_params: Additional CatBoost parameters
        """
        self.quantiles = sorted(quantiles)  # Ensure quantiles are sorted
        self.random_state = random_state
        self.catboost_params = catboost_params
        
        # Set up multi-quantile loss function
        alpha_str = ",".join([str(q) for q in self.quantiles])
        self.catboost_params["loss_function"] = f"MultiQuantile:alpha={alpha_str}"
        self.catboost_params["random_state"] = random_state
        
        # Initialize model
        self._model: Optional[CatBoostRegressor] = None
        self._feature_names: Optional[List[str]] = None
        self._is_fitted = False
        
    def fit(
        self,
        train_pool: Pool,
        eval_pool: Optional[Pool] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False,
        **fit_params: Any
    ) -> "CatBoostMultiQuantileModel":
        """
        Fit the CatBoost model.
        
        Args:
            train_pool: Training data as CatBoost Pool
            eval_pool: Evaluation data as CatBoost Pool (optional)
            early_stopping_rounds: Early stopping rounds
            verbose: Whether to show training progress
            **fit_params: Additional parameters for model.fit()
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting CatBoost multi-quantile model...")
        
        # Create model with parameters
        self._model = CatBoostRegressor(**self.catboost_params)
        
        # Set up fit parameters
        fit_kwargs = {
            "verbose": verbose,
            "plot": False,
            **fit_params
        }
        
        if eval_pool is not None:
            fit_kwargs["eval_set"] = eval_pool
            
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            
        # Fit the model
        self._model.fit(train_pool, **fit_kwargs)
        
        # Store feature names for validation
        self._feature_names = train_pool.get_feature_names()
        self._is_fitted = True
        
        logger.info(f"Model fitted successfully. Best iteration: {self.best_iteration}")
        return self
        
    def predict(
        self, 
        X: Union[pd.DataFrame, Pool, np.ndarray],
        ensure_monotonic: bool = True
    ) -> np.ndarray:
        """
        Generate multi-quantile predictions.
        
        Args:
            X: Input features (DataFrame, Pool, or array)
            ensure_monotonic: Whether to enforce monotonic quantile ordering
            
        Returns:
            Predictions array of shape (n_samples, n_quantiles)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if self._model is None:
            raise ValueError("Model is not initialized")
            
        # Convert to Pool if needed
        if isinstance(X, pd.DataFrame):
            # Validate feature names if available
            if self._feature_names is not None:
                missing_features = set(self._feature_names) - set(X.columns)
                if missing_features:
                    raise ValueError(f"Missing features in input data: {missing_features}")
                    
                # Reorder columns to match training
                X = X[self._feature_names]
                
            # Detect categorical features
            cat_features = np.where((X.dtypes == "category") | (X.dtypes == "object"))[0]
            pool = Pool(X, cat_features=cat_features)
        elif isinstance(X, Pool):
            pool = X
        else:
            # Assume numpy array
            pool = Pool(X)
            
        # Make predictions
        predictions = self._model.predict(pool)
        predictions = np.asarray(predictions)
        
        # Ensure proper shape
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
            
        # Enforce monotonic quantile ordering if requested
        if ensure_monotonic:
            predictions.sort(axis=1)
            
        return predictions
        
    def predict_quantiles(
        self, 
        X: Union[pd.DataFrame, Pool, np.ndarray],
        quantile_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Predict specific quantiles by index.
        
        Args:
            X: Input features
            quantile_indices: Indices of quantiles to return (default: all)
            
        Returns:
            Predictions for specified quantiles
        """
        all_predictions = self.predict(X)
        
        if quantile_indices is None:
            return all_predictions
            
        return all_predictions[:, quantile_indices]
        
    @property 
    def best_iteration(self) -> int:
        """Get the best iteration from training."""
        if self._model is None:
            return 0
        return getattr(self._model, 'best_iteration_', 0)
        
    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Get feature importances from the trained model."""
        if not self._is_fitted or self._model is None:
            return None
        return self._model.feature_importances_
        
    @property
    def feature_names(self) -> Optional[List[str]]:
        """Get feature names used during training."""
        return self._feature_names
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "quantiles": self.quantiles,
            "random_state": self.random_state,
            **self.catboost_params
        }
        
    def set_params(self, **params: Any) -> "CatBoostMultiQuantileModel":
        """Set model parameters."""
        if "quantiles" in params:
            self.quantiles = sorted(params.pop("quantiles"))
            alpha_str = ",".join([str(q) for q in self.quantiles])
            params["loss_function"] = f"MultiQuantile:alpha={alpha_str}"
            
        if "random_state" in params:
            self.random_state = params["random_state"]
            
        self.catboost_params.update(params)
        
        # Reset model if parameters changed
        self._model = None
        self._is_fitted = False
        
        return self
        
    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self._is_fitted or self._model is None:
            raise ValueError("Model must be fitted before saving")
            
        self._model.save_model(path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        self._model = CatBoostRegressor()
        self._model.load_model(path)
        self._is_fitted = True
        logger.info(f"Model loaded from {path}")
        
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self._is_fitted else "not fitted"
        return f"CatBoostMultiQuantileModel(quantiles={self.quantiles}, {status})"
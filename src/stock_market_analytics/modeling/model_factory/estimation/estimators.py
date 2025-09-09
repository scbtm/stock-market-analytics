"""
ML estimator wrappers for sklearn pipeline compatibility.

This module contains estimator classes that wrap various ML models
to ensure they work seamlessly with sklearn pipelines while providing
additional functionality for financial modeling.
"""

from typing import Any, Dict, List

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from ..protocols import SklearnCompatibleEstimator, QuantileEstimator
from .estimation_functions import (
    create_catboost_pool,
    prepare_features_for_sklearn,
    feature_importance_to_dict,
    handle_missing_values,
    validate_feature_matrix,
)


class CatBoostQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    CatBoost wrapper for quantile regression with Polars DataFrame support.
    
    Provides sklearn-compatible interface while handling CatBoost-specific
    functionality like categorical features and quantile loss.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        categorical_features: List[str] | None = None,
        **catboost_params: Any
    ):
        """
        Initialize the CatBoost quantile regressor.
        
        Args:
            quantiles: List of quantiles to predict
            categorical_features: List of categorical feature names
            **catboost_params: Additional CatBoost parameters
        """
        self.quantiles = sorted(quantiles)
        self.categorical_features = categorical_features or []
        self.catboost_params = catboost_params
        self.model_: Any = None
        self.feature_names_: List[str] = []
        
    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> "CatBoostQuantileRegressor":
        """
        Fit the CatBoost quantile regressor.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional fit parameters (e.g., eval_set, early_stopping_rounds)
            
        Returns:
            Self for method chaining
        """
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("CatBoost is required for this estimator")
        
        self.feature_names_ = X.columns
        
        # Set up quantile loss
        alpha_str = ",".join([str(q) for q in self.quantiles])
        default_params = {
            'loss_function': f'MultiQuantile:alpha={alpha_str}',
            'verbose': False,
            'random_state': 42
        }
        
        # Merge with user parameters
        params = {**default_params, **self.catboost_params}
        
        # Initialize model
        self.model_ = CatBoostRegressor(**params)
        
        # Create CatBoost Pool
        train_pool = create_catboost_pool(X, y, self.categorical_features)
        
        # Handle evaluation set if provided
        eval_set = kwargs.get('eval_set')
        eval_pool = None
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_pool = create_catboost_pool(X_val, y_val, self.categorical_features)
            kwargs['eval_set'] = eval_pool
        
        # Fit model
        self.model_.fit(train_pool, **kwargs)
        
        return self
    
    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        Make quantile predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Quantile predictions with shape (n_samples, n_quantiles)
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create prediction pool
        pred_pool = create_catboost_pool(X, categorical_features=self.categorical_features)
        
        # Make predictions
        predictions = self.model_.predict(pred_pool)
        
        # Ensure correct shape
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, len(self.quantiles))
        
        return predictions
    
    def predict_quantiles(self, X: pl.DataFrame, quantiles: List[float]) -> np.ndarray:
        """
        Predict specific quantiles.
        
        Args:
            X: Features for prediction
            quantiles: List of quantiles to predict
            
        Returns:
            Quantile predictions for specified quantiles
        """
        all_predictions = self.predict(X)
        
        # Map requested quantiles to model quantiles
        quantile_indices = []
        for q in quantiles:
            if q in self.quantiles:
                quantile_indices.append(self.quantiles.index(q))
            else:
                # Find closest quantile
                closest_idx = np.argmin([abs(q - mq) for mq in self.quantiles])
                quantile_indices.append(closest_idx)
        
        return all_predictions[:, quantile_indices]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_values = self.model_.get_feature_importance()
        return feature_importance_to_dict(self.feature_names_, importance_values)


class LightGBMQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    LightGBM wrapper for quantile regression with Polars DataFrame support.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        categorical_features: List[str] | None = None,
        **lgb_params: Any
    ):
        """
        Initialize the LightGBM quantile regressor.
        
        Args:
            quantiles: List of quantiles to predict
            categorical_features: List of categorical feature names
            **lgb_params: Additional LightGBM parameters
        """
        self.quantiles = sorted(quantiles)
        self.categorical_features = categorical_features or []
        self.lgb_params = lgb_params
        self.models_: Dict[float, Any] = {}
        self.feature_names_: List[str] = []
        
    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> "LightGBMQuantileRegressor":
        """
        Fit separate LightGBM models for each quantile.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional fit parameters
            
        Returns:
            Self for method chaining
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required for this estimator")
        
        self.feature_names_ = X.columns
        
        # Prepare data
        X_array, feature_names = prepare_features_for_sklearn(X)
        y_array = y.to_numpy()
        
        # Train a model for each quantile
        for quantile in self.quantiles:
            # Set quantile-specific parameters
            params = {
                'objective': 'quantile',
                'alpha': quantile,
                'metric': 'quantile',
                'verbose': -1,
                **self.lgb_params
            }
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(
                X_array, 
                label=y_array,
                feature_name=feature_names,
                categorical_feature=self.categorical_features
            )
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                **kwargs
            )
            
            self.models_[quantile] = model
        
        return self
    
    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        Make quantile predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Quantile predictions with shape (n_samples, n_quantiles)
        """
        if not self.models_:
            raise ValueError("Models must be fitted before making predictions")
        
        X_array, _ = prepare_features_for_sklearn(X)
        
        predictions = []
        for quantile in self.quantiles:
            pred = self.models_[quantile].predict(X_array)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def predict_quantiles(self, X: pl.DataFrame, quantiles: List[float]) -> np.ndarray:
        """
        Predict specific quantiles.
        
        Args:
            X: Features for prediction
            quantiles: List of quantiles to predict
            
        Returns:
            Quantile predictions for specified quantiles
        """
        X_array, _ = prepare_features_for_sklearn(X)
        
        predictions = []
        for quantile in quantiles:
            if quantile in self.models_:
                pred = self.models_[quantile].predict(X_array)
            else:
                # Interpolate between available quantiles
                lower_q = max([q for q in self.quantiles if q <= quantile], default=min(self.quantiles))
                upper_q = min([q for q in self.quantiles if q >= quantile], default=max(self.quantiles))
                
                if lower_q == upper_q:
                    pred = self.models_[lower_q].predict(X_array)
                else:
                    pred_lower = self.models_[lower_q].predict(X_array)
                    pred_upper = self.models_[upper_q].predict(X_array)
                    
                    # Linear interpolation
                    weight = (quantile - lower_q) / (upper_q - lower_q)
                    pred = pred_lower + weight * (pred_upper - pred_lower)
            
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (averaged across quantile models)."""
        if not self.models_:
            raise ValueError("Models must be fitted before getting feature importance")
        
        # Average importance across all quantile models
        all_importance = {}
        for model in self.models_.values():
            importance = model.feature_importance(importance_type='gain')
            for i, feat_name in enumerate(self.feature_names_):
                if feat_name not in all_importance:
                    all_importance[feat_name] = []
                all_importance[feat_name].append(importance[i])
        
        # Calculate mean importance
        mean_importance = {
            feat: float(np.mean(scores)) 
            for feat, scores in all_importance.items()
        }
        
        return mean_importance


class RandomForestQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Random Forest wrapper for quantile regression using quantile forest approach.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        **rf_params: Any
    ):
        """
        Initialize the Random Forest quantile regressor.
        
        Args:
            quantiles: List of quantiles to predict
            **rf_params: Random Forest parameters
        """
        self.quantiles = sorted(quantiles)
        self.rf_params = rf_params
        self.model_: Any = None
        self.feature_names_: List[str] = []
        
    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> "RandomForestQuantileRegressor":
        """
        Fit the Random Forest quantile regressor.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional fit parameters
            
        Returns:
            Self for method chaining
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            raise ImportError("scikit-learn is required for this estimator")
        
        self.feature_names_ = X.columns
        
        # Prepare data
        X_array, _ = prepare_features_for_sklearn(X)
        y_array = y.to_numpy()
        
        # Default parameters for quantile estimation
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
        
        params = {**default_params, **self.rf_params}
        
        # Fit Random Forest
        self.model_ = RandomForestRegressor(**params)
        self.model_.fit(X_array, y_array)
        
        return self
    
    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        Make quantile predictions using tree predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Quantile predictions with shape (n_samples, n_quantiles)
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_array, _ = prepare_features_for_sklearn(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X_array) for tree in self.model_.estimators_
        ]).T  # Shape: (n_samples, n_trees)
        
        # Calculate quantiles across trees for each sample
        quantile_predictions = np.percentile(
            tree_predictions, 
            [q * 100 for q in self.quantiles], 
            axis=1
        ).T  # Shape: (n_samples, n_quantiles)
        
        return quantile_predictions
    
    def predict_quantiles(self, X: pl.DataFrame, quantiles: List[float]) -> np.ndarray:
        """
        Predict specific quantiles.
        
        Args:
            X: Features for prediction
            quantiles: List of quantiles to predict
            
        Returns:
            Quantile predictions for specified quantiles
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_array, _ = prepare_features_for_sklearn(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X_array) for tree in self.model_.estimators_
        ]).T
        
        # Calculate requested quantiles
        quantile_predictions = np.percentile(
            tree_predictions, 
            [q * 100 for q in quantiles], 
            axis=1
        ).T
        
        return quantile_predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_values = self.model_.feature_importances_
        return feature_importance_to_dict(self.feature_names_, importance_values)


class LinearQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Linear quantile regression using scikit-learn's QuantileRegressor.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        **quantile_params: Any
    ):
        """
        Initialize the Linear quantile regressor.
        
        Args:
            quantiles: List of quantiles to predict
            **quantile_params: QuantileRegressor parameters
        """
        self.quantiles = sorted(quantiles)
        self.quantile_params = quantile_params
        self.models_: Dict[float, Any] = {}
        self.feature_names_: List[str] = []
        
    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> "LinearQuantileRegressor":
        """
        Fit linear quantile regression models.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional fit parameters
            
        Returns:
            Self for method chaining
        """
        try:
            from sklearn.linear_model import QuantileRegressor
        except ImportError:
            raise ImportError("scikit-learn >= 0.24 is required for QuantileRegressor")
        
        self.feature_names_ = X.columns
        
        # Prepare data
        X_array, _ = prepare_features_for_sklearn(X)
        y_array = y.to_numpy()
        
        # Fit a model for each quantile
        for quantile in self.quantiles:
            model = QuantileRegressor(quantile=quantile, **self.quantile_params)
            model.fit(X_array, y_array)
            self.models_[quantile] = model
        
        return self
    
    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        Make quantile predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Quantile predictions with shape (n_samples, n_quantiles)
        """
        if not self.models_:
            raise ValueError("Models must be fitted before making predictions")
        
        X_array, _ = prepare_features_for_sklearn(X)
        
        predictions = []
        for quantile in self.quantiles:
            pred = self.models_[quantile].predict(X_array)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def predict_quantiles(self, X: pl.DataFrame, quantiles: List[float]) -> np.ndarray:
        """
        Predict specific quantiles.
        
        Args:
            X: Features for prediction
            quantiles: List of quantiles to predict
            
        Returns:
            Quantile predictions for specified quantiles
        """
        X_array, _ = prepare_features_for_sklearn(X)
        
        predictions = []
        for quantile in quantiles:
            if quantile in self.models_:
                pred = self.models_[quantile].predict(X_array)
            else:
                # Train model for new quantile
                y_dummy = np.zeros(X_array.shape[0])  # Placeholder
                model = QuantileRegressor(quantile=quantile, **self.quantile_params)
                # Note: This would require access to training data, which we don't have here
                # In practice, you'd want to either pre-fit all needed quantiles or 
                # store training data for on-demand fitting
                raise ValueError(f"Quantile {quantile} was not fitted. Pre-fit all required quantiles.")
            
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on coefficient magnitudes."""
        if not self.models_:
            raise ValueError("Models must be fitted before getting feature importance")
        
        # Average coefficient magnitudes across quantile models
        all_coefs = []
        for model in self.models_.values():
            all_coefs.append(np.abs(model.coef_))
        
        mean_coefs = np.mean(all_coefs, axis=0)
        return feature_importance_to_dict(self.feature_names_, mean_coefs)


class BaselineEstimator(BaseEstimator, RegressorMixin):
    """
    Simple baseline estimator for comparison purposes.
    
    Provides basic statistical baselines like mean, median, or simple trend models.
    """
    
    def __init__(self, strategy: str = "mean"):
        """
        Initialize the baseline estimator.
        
        Args:
            strategy: Baseline strategy ("mean", "median", "last", "trend")
        """
        self.strategy = strategy
        self.baseline_value_: float = 0.0
        self.trend_coef_: float = 0.0
        self.feature_names_: List[str] = []
        
    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> "BaselineEstimator":
        """
        Fit the baseline estimator.
        
        Args:
            X: Training features (may not be used depending on strategy)
            y: Training targets
            **kwargs: Additional fit parameters
            
        Returns:
            Self for method chaining
        """
        self.feature_names_ = X.columns
        y_array = y.to_numpy()
        
        if self.strategy == "mean":
            self.baseline_value_ = float(np.mean(y_array))
        elif self.strategy == "median":
            self.baseline_value_ = float(np.median(y_array))
        elif self.strategy == "last":
            self.baseline_value_ = float(y_array[-1])
        elif self.strategy == "trend":
            # Simple linear trend
            time_index = np.arange(len(y_array))
            self.trend_coef_ = float(np.polyfit(time_index, y_array, 1)[0])
            self.baseline_value_ = float(y_array[-1])
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return self
    
    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        Make baseline predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Baseline predictions
        """
        n_samples = X.height
        
        if self.strategy in ["mean", "median", "last"]:
            predictions = np.full(n_samples, self.baseline_value_)
        elif self.strategy == "trend":
            # Project trend forward
            predictions = np.array([
                self.baseline_value_ + self.trend_coef_ * i 
                for i in range(1, n_samples + 1)
            ])
        else:
            predictions = np.zeros(n_samples)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (all zeros for baseline)."""
        return {feat: 0.0 for feat in self.feature_names_}
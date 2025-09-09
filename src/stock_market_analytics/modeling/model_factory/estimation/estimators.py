"""
ML estimator wrappers for sklearn pipeline compatibility.

This module contains estimator classes that wrap various ML models
to ensure they work seamlessly with sklearn pipelines while providing
additional functionality for financial modeling.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array
from sklearn.metrics import r2_score


from stock_market_analytics.modeling.model_factory.estimation.estimation_functions import (
    detect_categorical_features,
    create_catboost_pool,
)

from catboost import CatBoostRegressor


class CatBoostMultiQuantileModel(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for CatBoost multi-quantile regression.

    Parameters
    ----------
    quantiles : list of float
        Quantiles to predict, e.g., [0.1, 0.5, 0.9]
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to display training progress
    **catboost_params
        Additional CatBoost parameters
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        random_state: int = 1,
        verbose: bool = False,
        **catboost_params: Any,
    ):
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.random_state = random_state
        self.verbose = verbose
        self.catboost_params = catboost_params
        self.cat_features_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        **fit_params: Any,
    ) -> "CatBoostMultiQuantileModel":
        """Fit the CatBoost multi-quantile model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target values
        **fit_params
            Additional parameters for CatBoost.fit()

        Returns
        -------
        self : CatBoostMultiQuantileModel
            Fitted estimator
        """
        #X, y = check_X_y(X, y, accept_sparse=False, dtype=None)

        # Build CatBoost parameters
        params = self.catboost_params.copy()

        # Set up multi-quantile loss
        sorted_quantiles = sorted(self.quantiles)
        alpha_str = ",".join([str(q) for q in sorted_quantiles])
        params["loss_function"] = f"MultiQuantile:alpha={alpha_str}"
        params["random_state"] = self.random_state
        params["verbose"] = self.verbose
        params["use_best_model"] = True if "eval_set" in fit_params else False  # Enable best model tracking

        # Create and fit model
        self._model = CatBoostRegressor(**params)

        train_pool = create_catboost_pool(X, y)
        if "eval_set" in fit_params:
            X_val, y_val = fit_params["eval_set"]
            eval_pool = create_catboost_pool(X_val, y_val)
            fit_params["eval_set"] = eval_pool
        self._model.fit(train_pool, **fit_params)

        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Generate multi-quantile predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features

        Returns
        -------
        predictions : ndarray of shape (n_samples, n_quantiles)
            Multi-quantile predictions
        """
        if self._model is None:
            raise ValueError("Model must be fitted before making predictions")

        pool = create_catboost_pool(X)
        predictions = self._model.predict(pool)
        predictions = np.asarray(predictions)

        # Ensure proper shape for multi-quantile output
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        # Ensure monotonic quantile ordering
        predictions.sort(axis=1)

        return predictions

    def transform(self, X: Any) -> np.ndarray:
        """Alias for predict to support transformer interface inside pipelines."""
        return self.predict(X)

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return the coefficient of determination R^2 for median quantile.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values for X
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights

        Returns
        -------
        score : float
            R^2 score for median quantile prediction
        """
        predictions = self.predict(X)

        # Use median quantile (middle column) for scoring
        median_idx = len(self.quantiles) // 2
        median_pred = predictions[:, median_idx]

        return r2_score(y, median_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        """Feature importances from the trained model."""
        if self._model is None:
            raise ValueError("Model must be fitted to access feature importances")
        return self._model.feature_importances_

    @property
    def best_iteration_(self):
        """Best iteration from training with early stopping."""
        if self._model is None:
            raise ValueError("Model must be fitted to access best iteration")
        return getattr(self._model, "best_iteration_", None)

    def get_params(self, _deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            "quantiles": self.quantiles,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }
        params.update(self.catboost_params)
        return params

    def set_params(self, **params: Any) -> "CatBoostMultiQuantileModel":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : CatBoostMultiQuantileModel
            Estimator instance.
        """
        for key, value in params.items():
            if key in ["quantiles", "random_state", "verbose"]:
                setattr(self, key, value)
            else:
                # Assume it's a CatBoost parameter
                self.catboost_params[key] = value

        return self


# class LightGBMQuantileRegressor(BaseEstimator, RegressorMixin):
#     """
#     LightGBM wrapper for quantile regression with Polars DataFrame support.
#     """

#     def __init__(
#         self,
#         quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
#         categorical_features: List[str] | None = None,
#         **lgb_params: Any
#     ):
#         """
#         Initialize the LightGBM quantile regressor.

#         Args:
#             quantiles: List of quantiles to predict
#             categorical_features: List of categorical feature names
#             **lgb_params: Additional LightGBM parameters
#         """
#         self.quantiles = sorted(quantiles)
#         self.categorical_features = categorical_features or []
#         self.lgb_params = lgb_params
#         self.models_: Dict[float, Any] = {}
#         self.feature_names_: List[str] = []

#     def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> "LightGBMQuantileRegressor":
#         """
#         Fit separate LightGBM models for each quantile.

#         Args:
#             X: Training features
#             y: Training targets
#             **kwargs: Additional fit parameters

#         Returns:
#             Self for method chaining
#         """
#         try:
#             import lightgbm as lgb
#         except ImportError:
#             raise ImportError("LightGBM is required for this estimator")

#         self.feature_names_ = X.columns

#         # Prepare data
#         X_array, feature_names = prepare_features_for_sklearn(X)
#         y_array = y.to_numpy()

#         # Train a model for each quantile
#         for quantile in self.quantiles:
#             # Set quantile-specific parameters
#             params = {
#                 'objective': 'quantile',
#                 'alpha': quantile,
#                 'metric': 'quantile',
#                 'verbose': -1,
#                 **self.lgb_params
#             }

#             # Create LightGBM datasets
#             train_data = lgb.Dataset(
#                 X_array,
#                 label=y_array,
#                 feature_name=feature_names,
#                 categorical_feature=self.categorical_features
#             )

#             # Train model
#             model = lgb.train(
#                 params,
#                 train_data,
#                 **kwargs
#             )

#             self.models_[quantile] = model

#         return self

#     def predict(self, X: pl.DataFrame) -> np.ndarray:
#         """
#         Make quantile predictions.

#         Args:
#             X: Features for prediction

#         Returns:
#             Quantile predictions with shape (n_samples, n_quantiles)
#         """
#         if not self.models_:
#             raise ValueError("Models must be fitted before making predictions")

#         X_array, _ = prepare_features_for_sklearn(X)

#         predictions = []
#         for quantile in self.quantiles:
#             pred = self.models_[quantile].predict(X_array)
#             predictions.append(pred)

#         return np.column_stack(predictions)

#     def predict_quantiles(self, X: pl.DataFrame, quantiles: List[float]) -> np.ndarray:
#         """
#         Predict specific quantiles.

#         Args:
#             X: Features for prediction
#             quantiles: List of quantiles to predict

#         Returns:
#             Quantile predictions for specified quantiles
#         """
#         X_array, _ = prepare_features_for_sklearn(X)

#         predictions = []
#         for quantile in quantiles:
#             if quantile in self.models_:
#                 pred = self.models_[quantile].predict(X_array)
#             else:
#                 # Interpolate between available quantiles
#                 lower_q = max([q for q in self.quantiles if q <= quantile], default=min(self.quantiles))
#                 upper_q = min([q for q in self.quantiles if q >= quantile], default=max(self.quantiles))

#                 if lower_q == upper_q:
#                     pred = self.models_[lower_q].predict(X_array)
#                 else:
#                     pred_lower = self.models_[lower_q].predict(X_array)
#                     pred_upper = self.models_[upper_q].predict(X_array)

#                     # Linear interpolation
#                     weight = (quantile - lower_q) / (upper_q - lower_q)
#                     pred = pred_lower + weight * (pred_upper - pred_lower)

#             predictions.append(pred)

#         return np.column_stack(predictions)

#     def get_feature_importance(self) -> Dict[str, float]:
#         """Get feature importance scores (averaged across quantile models)."""
#         if not self.models_:
#             raise ValueError("Models must be fitted before getting feature importance")

#         # Average importance across all quantile models
#         all_importance = {}
#         for model in self.models_.values():
#             importance = model.feature_importance(importance_type='gain')
#             for i, feat_name in enumerate(self.feature_names_):
#                 if feat_name not in all_importance:
#                     all_importance[feat_name] = []
#                 all_importance[feat_name].append(importance[i])

#         # Calculate mean importance
#         mean_importance = {
#             feat: float(np.mean(scores))
#             for feat, scores in all_importance.items()
#         }

#         return mean_importance


# class RandomForestQuantileRegressor(BaseEstimator, RegressorMixin):
#     """
#     Random Forest wrapper for quantile regression using quantile forest approach.
#     """

#     def __init__(
#         self,
#         quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
#         **rf_params: Any
#     ):
#         """
#         Initialize the Random Forest quantile regressor.

#         Args:
#             quantiles: List of quantiles to predict
#             **rf_params: Random Forest parameters
#         """
#         self.quantiles = sorted(quantiles)
#         self.rf_params = rf_params
#         self.model_: Any = None
#         self.feature_names_: List[str] = []

#     def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> "RandomForestQuantileRegressor":
#         """
#         Fit the Random Forest quantile regressor.

#         Args:
#             X: Training features
#             y: Training targets
#             **kwargs: Additional fit parameters

#         Returns:
#             Self for method chaining
#         """
#         try:
#             from sklearn.ensemble import RandomForestRegressor
#         except ImportError:
#             raise ImportError("scikit-learn is required for this estimator")

#         self.feature_names_ = X.columns

#         # Prepare data
#         X_array, _ = prepare_features_for_sklearn(X)
#         y_array = y.to_numpy()

#         # Default parameters for quantile estimation
#         default_params = {
#             'n_estimators': 100,
#             'random_state': 42,
#             'n_jobs': -1
#         }

#         params = {**default_params, **self.rf_params}

#         # Fit Random Forest
#         self.model_ = RandomForestRegressor(**params)
#         self.model_.fit(X_array, y_array)

#         return self

#     def predict(self, X: pl.DataFrame) -> np.ndarray:
#         """
#         Make quantile predictions using tree predictions.

#         Args:
#             X: Features for prediction

#         Returns:
#             Quantile predictions with shape (n_samples, n_quantiles)
#         """
#         if self.model_ is None:
#             raise ValueError("Model must be fitted before making predictions")

#         X_array, _ = prepare_features_for_sklearn(X)

#         # Get predictions from all trees
#         tree_predictions = np.array([
#             tree.predict(X_array) for tree in self.model_.estimators_
#         ]).T  # Shape: (n_samples, n_trees)

#         # Calculate quantiles across trees for each sample
#         quantile_predictions = np.percentile(
#             tree_predictions,
#             [q * 100 for q in self.quantiles],
#             axis=1
#         ).T  # Shape: (n_samples, n_quantiles)

#         return quantile_predictions

#     def predict_quantiles(self, X: pl.DataFrame, quantiles: List[float]) -> np.ndarray:
#         """
#         Predict specific quantiles.

#         Args:
#             X: Features for prediction
#             quantiles: List of quantiles to predict

#         Returns:
#             Quantile predictions for specified quantiles
#         """
#         if self.model_ is None:
#             raise ValueError("Model must be fitted before making predictions")

#         X_array, _ = prepare_features_for_sklearn(X)

#         # Get predictions from all trees
#         tree_predictions = np.array([
#             tree.predict(X_array) for tree in self.model_.estimators_
#         ]).T

#         # Calculate requested quantiles
#         quantile_predictions = np.percentile(
#             tree_predictions,
#             [q * 100 for q in quantiles],
#             axis=1
#         ).T

#         return quantile_predictions

#     def get_feature_importance(self) -> Dict[str, float]:
#         """Get feature importance scores."""
#         if self.model_ is None:
#             raise ValueError("Model must be fitted before getting feature importance")

#         importance_values = self.model_.feature_importances_
#         return feature_importance_to_dict(self.feature_names_, importance_values)


# class LinearQuantileRegressor(BaseEstimator, RegressorMixin):
#     """
#     Linear quantile regression using scikit-learn's QuantileRegressor.
#     """

#     def __init__(
#         self,
#         quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
#         **quantile_params: Any
#     ):
#         """
#         Initialize the Linear quantile regressor.

#         Args:
#             quantiles: List of quantiles to predict
#             **quantile_params: QuantileRegressor parameters
#         """
#         self.quantiles = sorted(quantiles)
#         self.quantile_params = quantile_params
#         self.models_: Dict[float, Any] = {}
#         self.feature_names_: List[str] = []

#     def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> "LinearQuantileRegressor":
#         """
#         Fit linear quantile regression models.

#         Args:
#             X: Training features
#             y: Training targets
#             **kwargs: Additional fit parameters

#         Returns:
#             Self for method chaining
#         """
#         try:
#             from sklearn.linear_model import QuantileRegressor
#         except ImportError:
#             raise ImportError("scikit-learn >= 0.24 is required for QuantileRegressor")

#         self.feature_names_ = X.columns

#         # Prepare data
#         X_array, _ = prepare_features_for_sklearn(X)
#         y_array = y.to_numpy()

#         # Fit a model for each quantile
#         for quantile in self.quantiles:
#             model = QuantileRegressor(quantile=quantile, **self.quantile_params)
#             model.fit(X_array, y_array)
#             self.models_[quantile] = model

#         return self

#     def predict(self, X: pl.DataFrame) -> np.ndarray:
#         """
#         Make quantile predictions.

#         Args:
#             X: Features for prediction

#         Returns:
#             Quantile predictions with shape (n_samples, n_quantiles)
#         """
#         if not self.models_:
#             raise ValueError("Models must be fitted before making predictions")

#         X_array, _ = prepare_features_for_sklearn(X)

#         predictions = []
#         for quantile in self.quantiles:
#             pred = self.models_[quantile].predict(X_array)
#             predictions.append(pred)

#         return np.column_stack(predictions)

#     def predict_quantiles(self, X: pl.DataFrame, quantiles: List[float]) -> np.ndarray:
#         """
#         Predict specific quantiles.

#         Args:
#             X: Features for prediction
#             quantiles: List of quantiles to predict

#         Returns:
#             Quantile predictions for specified quantiles
#         """
#         X_array, _ = prepare_features_for_sklearn(X)

#         predictions = []
#         for quantile in quantiles:
#             if quantile in self.models_:
#                 pred = self.models_[quantile].predict(X_array)
#             else:
#                 # Train model for new quantile
#                 y_dummy = np.zeros(X_array.shape[0])  # Placeholder
#                 model = QuantileRegressor(quantile=quantile, **self.quantile_params)
#                 # Note: This would require access to training data, which we don't have here
#                 # In practice, you'd want to either pre-fit all needed quantiles or
#                 # store training data for on-demand fitting
#                 raise ValueError(f"Quantile {quantile} was not fitted. Pre-fit all required quantiles.")

#             predictions.append(pred)

#         return np.column_stack(predictions)

#     def get_feature_importance(self) -> Dict[str, float]:
#         """Get feature importance based on coefficient magnitudes."""
#         if not self.models_:
#             raise ValueError("Models must be fitted before getting feature importance")

#         # Average coefficient magnitudes across quantile models
#         all_coefs = []
#         for model in self.models_.values():
#             all_coefs.append(np.abs(model.coef_))

#         mean_coefs = np.mean(all_coefs, axis=0)
#         return feature_importance_to_dict(self.feature_names_, mean_coefs)


# class BaselineEstimator(BaseEstimator, RegressorMixin):
#     """
#     Simple baseline estimator for comparison purposes.

#     Provides basic statistical baselines like mean, median, or simple trend models.
#     """

#     def __init__(self, strategy: str = "mean"):
#         """
#         Initialize the baseline estimator.

#         Args:
#             strategy: Baseline strategy ("mean", "median", "last", "trend")
#         """
#         self.strategy = strategy
#         self.baseline_value_: float = 0.0
#         self.trend_coef_: float = 0.0
#         self.feature_names_: List[str] = []

#     def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> "BaselineEstimator":
#         """
#         Fit the baseline estimator.

#         Args:
#             X: Training features (may not be used depending on strategy)
#             y: Training targets
#             **kwargs: Additional fit parameters

#         Returns:
#             Self for method chaining
#         """
#         self.feature_names_ = X.columns
#         y_array = y.to_numpy()

#         if self.strategy == "mean":
#             self.baseline_value_ = float(np.mean(y_array))
#         elif self.strategy == "median":
#             self.baseline_value_ = float(np.median(y_array))
#         elif self.strategy == "last":
#             self.baseline_value_ = float(y_array[-1])
#         elif self.strategy == "trend":
#             # Simple linear trend
#             time_index = np.arange(len(y_array))
#             self.trend_coef_ = float(np.polyfit(time_index, y_array, 1)[0])
#             self.baseline_value_ = float(y_array[-1])
#         else:
#             raise ValueError(f"Unknown strategy: {self.strategy}")

#         return self

#     def predict(self, X: pl.DataFrame) -> np.ndarray:
#         """
#         Make baseline predictions.

#         Args:
#             X: Features for prediction

#         Returns:
#             Baseline predictions
#         """
#         n_samples = X.height

#         if self.strategy in ["mean", "median", "last"]:
#             predictions = np.full(n_samples, self.baseline_value_)
#         elif self.strategy == "trend":
#             # Project trend forward
#             predictions = np.array([
#                 self.baseline_value_ + self.trend_coef_ * i
#                 for i in range(1, n_samples + 1)
#             ])
#         else:
#             predictions = np.zeros(n_samples)

#         return predictions

#     def get_feature_importance(self) -> Dict[str, float]:
#         """Get feature importance (all zeros for baseline)."""
#         return {feat: 0.0 for feat in self.feature_names_}

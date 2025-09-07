"""Naive baseline quantile regressors for comparison with advanced models."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y

# from sklearn.linear_model import QuantileRegressor
# from sklearn.preprocessing import StandardScaler


class HistoricalQuantileBaseline(BaseEstimator, RegressorMixin):
    """
    Naive baseline that predicts historical quantiles from training data.

    This is the simplest possible quantile regressor - it ignores features
    and always predicts the empirical quantiles computed from training targets.

    Parameters
    ----------
    quantiles : list of float
        Quantiles to predict, e.g., [0.1, 0.5, 0.9]
    """

    def __init__(self, quantiles: list[float] | None = None):
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "HistoricalQuantileBaseline":
        _ = fit_params  # Unused but required for sklearn compatibility
        """
        Fit the baseline by computing empirical quantiles from training targets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features (ignored)
        y : array-like of shape (n_samples,)
            Training targets

        Returns
        -------
        self : HistoricalQuantileBaseline
        """
        X, y = check_X_y(X, y, accept_sparse=False)

        # Compute empirical quantiles from training data
        self.empirical_quantiles_ = np.quantile(y, self.quantiles)

        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()

        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict quantiles by repeating empirical quantiles for all samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features

        Returns
        -------
        predictions : ndarray of shape (n_samples, n_quantiles)
            Quantile predictions (same for all samples)
        """
        X = check_array(X, accept_sparse=False)

        if not hasattr(self, "empirical_quantiles_"):
            raise ValueError("Model must be fitted before making predictions")

        n_samples = X.shape[0]
        # Repeat the same quantiles for all samples
        predictions = np.tile(self.empirical_quantiles_, (n_samples, 1))

        return predictions

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return R^2 score for median quantile prediction."""
        from sklearn.metrics import r2_score

        predictions = self.predict(X)
        median_idx = len(self.quantiles) // 2
        median_pred = predictions[:, median_idx]

        return r2_score(y, median_pred, sample_weight=sample_weight)


# class LinearQuantileBaseline(BaseEstimator, RegressorMixin):
#     """
#     Simple linear quantile regression baseline using sklearn's QuantileRegressor.

#     Fits separate linear models for each quantile with optional feature scaling.

#     Parameters
#     ----------
#     quantiles : list of float
#         Quantiles to predict, e.g., [0.1, 0.5, 0.9]
#     alpha : float, default=0.0
#         L1 regularization parameter
#     scale_features : bool, default=True
#         Whether to standardize features
#     solver : str, default='highs'
#         Solver to use for QuantileRegressor
#     """

#     def __init__(
#         self,
#         quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
#         alpha: float = 0.0,
#         scale_features: bool = True,
#         solver: str = 'highs'
#     ):
#         self.quantiles = quantiles
#         self.alpha = alpha
#         self.scale_features = scale_features
#         self.solver = solver

#     def fit(self, X: Any, y: Any, **fit_params: Any) -> "LinearQuantileBaseline":
#         _ = fit_params  # Unused but required for sklearn compatibility
#         """
#         Fit separate linear quantile regressors for each quantile.

#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)
#             Training features
#         y : array-like of shape (n_samples,)
#             Training targets

#         Returns
#         -------
#         self : LinearQuantileBaseline
#         """
#         X, y = check_X_y(X, y, accept_sparse=False)

#         # Optional feature scaling
#         if self.scale_features:
#             self.scaler_ = StandardScaler()
#             X_scaled = self.scaler_.fit_transform(X)
#         else:
#             X_scaled = X
#             self.scaler_ = None

#         # Fit separate models for each quantile
#         self.models_ = {}
#         for quantile in self.quantiles:
#             model = QuantileRegressor(
#                 quantile=quantile,
#                 alpha=self.alpha,
#                 solver=self.solver
#             )
#             model.fit(X_scaled, y)
#             self.models_[quantile] = model

#         self.n_features_in_ = X.shape[1]
#         if hasattr(X, 'columns'):
#             self.feature_names_in_ = X.columns.tolist()

#         return self

#     def predict(self, X: Any) -> np.ndarray:
#         """
#         Generate quantile predictions using fitted linear models.

#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)
#             Input features

#         Returns
#         -------
#         predictions : ndarray of shape (n_samples, n_quantiles)
#             Multi-quantile predictions
#         """
#         X = check_array(X, accept_sparse=False)

#         if not hasattr(self, 'models_'):
#             raise ValueError("Model must be fitted before making predictions")

#         # Apply scaling if used during training
#         if self.scaler_ is not None:
#             X_scaled = self.scaler_.transform(X)
#         else:
#             X_scaled = X

#         # Predict with each quantile model
#         predictions = []
#         for quantile in self.quantiles:
#             pred = self.models_[quantile].predict(X_scaled)
#             predictions.append(pred)

#         # Stack predictions: shape (n_samples, n_quantiles)
#         predictions = np.column_stack(predictions)

#         # Ensure monotonic quantile ordering
#         predictions.sort(axis=1)

#         return predictions

#     def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
#         """Return R^2 score for median quantile prediction."""
#         from sklearn.metrics import r2_score

#         predictions = self.predict(X)
#         median_idx = len(self.quantiles) // 2
#         median_pred = predictions[:, median_idx]

#         return r2_score(y, median_pred, sample_weight=sample_weight)


# class MovingAverageQuantileBaseline(BaseEstimator, RegressorMixin):
#     """
#     Baseline using moving window quantiles from recent historical data.

#     For time series data, uses a rolling window of recent target values
#     to compute quantiles. Falls back to historical quantiles if insufficient data.

#     Parameters
#     ----------
#     quantiles : list of float
#         Quantiles to predict
#     window_size : int, default=252
#         Size of rolling window (e.g., 252 trading days for 1 year)
#     min_periods : int, default=30
#         Minimum periods required to compute rolling quantiles
#     """

#     def __init__(
#         self,
#         quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
#         window_size: int = 252,
#         min_periods: int = 30
#     ):
#         self.quantiles = quantiles
#         self.window_size = window_size
#         self.min_periods = min_periods

#     def fit(self, X: Any, y: Any, **fit_params: Any) -> "MovingAverageQuantileBaseline":
#         _ = fit_params  # Unused but required for sklearn compatibility
#         """
#         Store training data for computing rolling quantiles.

#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)
#             Training features (used for sample indexing)
#         y : array-like of shape (n_samples,)
#             Training targets

#         Returns
#         -------
#         self : MovingAverageQuantileBaseline
#         """
#         X, y = check_X_y(X, y, accept_sparse=False)

#         # Store training targets for rolling computation
#         self.y_train_ = y.copy()

#         # Compute fallback empirical quantiles
#         self.empirical_quantiles_ = np.quantile(y, self.quantiles)

#         self.n_features_in_ = X.shape[1]
#         if hasattr(X, 'columns'):
#             self.feature_names_in_ = X.columns.tolist()

#         return self

#     def predict(self, X: Any) -> np.ndarray:
#         """
#         Predict using rolling quantiles from recent training data.

#         For each prediction, uses the most recent window_size observations
#         to compute quantiles. Falls back to empirical quantiles if insufficient data.

#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)
#             Input features

#         Returns
#         -------
#         predictions : ndarray of shape (n_samples, n_quantiles)
#             Rolling quantile predictions
#         """
#         X = check_array(X, accept_sparse=False)

#         if not hasattr(self, 'y_train_'):
#             raise ValueError("Model must be fitted before making predictions")

#         n_samples = X.shape[0]
#         predictions = np.zeros((n_samples, len(self.quantiles)))

#         # For each prediction, use rolling window from training data
#         for i in range(n_samples):
#             # Use last window_size observations from training data
#             if len(self.y_train_) >= self.min_periods:
#                 window_data = self.y_train_[-self.window_size:]
#                 rolling_quantiles = np.quantile(window_data, self.quantiles)
#                 predictions[i] = rolling_quantiles
#             else:
#                 # Fallback to empirical quantiles if insufficient data
#                 predictions[i] = self.empirical_quantiles_

#         return predictions

#     def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
#         """Return R^2 score for median quantile prediction."""
#         from sklearn.metrics import r2_score

#         predictions = self.predict(X)
#         median_idx = len(self.quantiles) // 2
#         median_pred = predictions[:, median_idx]

#         return r2_score(y, median_pred, sample_weight=sample_weight)


# class NaiveQuantileEnsemble(BaseEstimator, RegressorMixin):
#     """
#     Ensemble of multiple naive baselines with simple averaging.

#     Combines predictions from HistoricalQuantile and LinearQuantile baselines.

#     Parameters
#     ----------
#     quantiles : list of float
#         Quantiles to predict
#     weights : list of float, optional
#         Weights for ensemble averaging [historical, linear]. Default: [0.3, 0.7]
#     """

#     def __init__(
#         self,
#         quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
#         weights: list[float] = None
#     ):
#         self.quantiles = quantiles
#         self.weights = weights or [0.3, 0.7]  # Favor linear model slightly

#     def fit(self, X: Any, y: Any, **fit_params: Any) -> "NaiveQuantileEnsemble":
#         _ = fit_params  # Unused but required for sklearn compatibility
#         """Fit both baseline models."""
#         X, y = check_X_y(X, y, accept_sparse=False)

#         # Initialize and fit component models
#         self.historical_model_ = HistoricalQuantileBaseline(quantiles=self.quantiles)
#         self.linear_model_ = LinearQuantileBaseline(quantiles=self.quantiles)

#         self.historical_model_.fit(X, y)
#         self.linear_model_.fit(X, y)

#         self.n_features_in_ = X.shape[1]
#         if hasattr(X, 'columns'):
#             self.feature_names_in_ = X.columns.tolist()

#         return self

#     def predict(self, X: Any) -> np.ndarray:
#         """Generate ensemble predictions by weighted averaging."""
#         X = check_array(X, accept_sparse=False)

#         if not hasattr(self, 'historical_model_'):
#             raise ValueError("Model must be fitted before making predictions")

#         # Get predictions from component models
#         historical_pred = self.historical_model_.predict(X)
#         linear_pred = self.linear_model_.predict(X)

#         # Weighted average
#         ensemble_pred = (self.weights[0] * historical_pred +
#                         self.weights[1] * linear_pred)

#         # Ensure monotonic quantile ordering
#         ensemble_pred.sort(axis=1)

#         return ensemble_pred

#     def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
#         """Return R^2 score for median quantile prediction."""
#         from sklearn.metrics import r2_score

#         predictions = self.predict(X)
#         median_idx = len(self.quantiles) // 2
#         median_pred = predictions[:, median_idx]

#         return r2_score(y, median_pred, sample_weight=sample_weight)

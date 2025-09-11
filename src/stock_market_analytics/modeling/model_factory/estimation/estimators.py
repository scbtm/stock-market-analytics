"""
ML estimator wrappers for sklearn pipeline compatibility.

This module contains estimator classes that wrap various ML models
to ensure they work seamlessly with sklearn pipelines while providing
additional functionality for financial modeling.
"""

from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from stock_market_analytics.modeling.model_factory.protocols import (
    QuantileEstimator, Array, Frame, Series
)


from stock_market_analytics.modeling.model_factory.estimation.estimation_functions import (
    create_catboost_pool,
)

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None


class CatBoostMultiQuantileModel(BaseEstimator, RegressorMixin, QuantileEstimator):
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
        quantiles: Sequence[float] | None = None,
        random_state: int = 1,
        verbose: bool = False,
        **catboost_params: Any,
    ):
        self.quantiles = list(quantiles) if quantiles else [0.1, 0.25, 0.5, 0.75, 0.9]
        self.random_state = random_state
        self.verbose = verbose
        self.catboost_params = catboost_params
        self.cat_features_: list[str] = []

    def fit(
        self,
        X: Frame | Array,
        y: Series | Array,
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
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not available. Please install it with 'pip install catboost'")
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

    def predict(self, X: Frame | Array, return_full_quantiles: bool = False) -> Array:
        """Generate multi-quantile predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features

        Returns
        -------
        median_pred : ndarray of shape (n_samples,)
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

        if return_full_quantiles:
            return predictions

        median_idx = len(self.quantiles) // 2
        median_pred = predictions[:, median_idx]

        return median_pred

    def transform(self, X: Frame | Array) -> Array:
        """Alias for predict to enable transformer usage in pipelines."""
        return self.predict(X)

    def score(self, X: Frame | Array, y: Series | Array, sample_weight: Any = None) -> float:
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
        median_pred = self.predict(X)

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

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

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
"""Any custom models used in the modeling pipeline must be defined here as a proper wrapper class,
to make them compatible with scikit-learn's API (fit, predict, get_params, set_params, etc)."""

#Scikit-learn compatible CatBoost multi-quantile regressor wrapper.

from typing import Any

import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y


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
        quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        random_state: int = 1,
        verbose: bool = False,
        **catboost_params: Any
    ):
        self.quantiles = quantiles
        self.random_state = random_state
        self.verbose = verbose
        self.catboost_params = catboost_params

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "CatBoostMultiQuantileModel":
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
        X, y = check_X_y(X, y, accept_sparse=False)

        # Build CatBoost parameters
        params = self.catboost_params.copy()

        # Set up multi-quantile loss
        sorted_quantiles = sorted(self.quantiles)
        alpha_str = ",".join([str(q) for q in sorted_quantiles])
        params["loss_function"] = f"MultiQuantile:alpha={alpha_str}"
        params["random_state"] = self.random_state
        params["verbose"] = self.verbose
        params["use_best_model"] = True  # Enable best model tracking

        # Create and fit model
        self._model = CatBoostRegressor(**params)

        # Detect categorical features if X is DataFrame
        cat_features = None
        if hasattr(X, 'dtypes'):
            cat_features = np.where((X.dtypes == "category") | (X.dtypes == "object"))[0]

        train_pool = Pool(X, y, cat_features=cat_features)
        self._model.fit(train_pool, **fit_params)

        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
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
        X = check_array(X, accept_sparse=False)

        if self._model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Detect categorical features if X is DataFrame
        cat_features = None
        if hasattr(X, 'dtypes'):
            cat_features = np.where((X.dtypes == "category") | (X.dtypes == "object"))[0]

        pool = Pool(X, cat_features=cat_features)
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
        from sklearn.metrics import r2_score

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
        return getattr(self._model, 'best_iteration_', None)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
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
            "verbose": self.verbose
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

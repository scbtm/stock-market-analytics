"""
ML estimator wrappers for sklearn pipeline compatibility.

This module contains estimator classes that wrap various ML models
to ensure they work seamlessly with sklearn pipelines while providing
additional functionality for financial modeling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

from stock_market_analytics.modeling.model_factory.estimation.estimation_functions import (
    create_catboost_pool,
)
from stock_market_analytics.modeling.model_factory.protocols import (
    Array,
    Frame,
    QuantileEstimator,
    Series,
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
    ) -> CatBoostMultiQuantileModel:
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
        # X, y = check_X_y(X, y, accept_sparse=False, dtype=None)

        # Build CatBoost parameters
        params = self.catboost_params.copy()

        # Set up multi-quantile loss
        sorted_quantiles = sorted(self.quantiles)
        alpha_str = ",".join([str(q) for q in sorted_quantiles])
        params["loss_function"] = f"MultiQuantile:alpha={alpha_str}"
        params["random_state"] = self.random_state
        params["verbose"] = self.verbose
        params["use_best_model"] = (
            "eval_set" in fit_params
        )  # Enable early stopping if eval_set is provided

        # Create and fit model
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost is not available. Please install it with 'pip install catboost'"
            )
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

    def score(
        self, X: Frame | Array, y: Series | Array, sample_weight: Any = None
    ) -> float:
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
    def feature_names_(self):
        """Feature names seen during fit."""
        return self._model.feature_names_ if self._model else None

    @property
    def best_iteration_(self):
        """Best iteration from training with early stopping."""
        if self._model is None:
            raise ValueError("Model must be fitted to access best iteration")
        return getattr(self._model, "best_iteration_", None)

    def get_params(self) -> dict[str, Any]:
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

    def set_params(self, **params: Any) -> CatBoostMultiQuantileModel:
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

    def get_feature_importance(self, **kwargs: dict) -> pd.DataFrame:
        """Get feature importances from the trained model.
        Parameters
        ----------
        **kwargs : dict
            Additional parameters for CatBoost's get_feature_importance method.
        Returns
        -------
        importances : Array
            Feature importances.
        """
        if self._model is None:
            raise ValueError("Model must be fitted to access feature importances")
        return self._model.get_feature_importance(**kwargs | {"prettified": True})  # type: ignore - this will always be a DF


# ---------------------------------------------------------------------------------------
# Baseline Estimator
# ---------------------------------------------------------------------------------------


class HistoricalMultiQuantileBaseline(BaseEstimator, RegressorMixin, QuantileEstimator):
    """
    Naive multi-quantile baseline that predicts unconditional (or per-group) empirical
    quantiles estimated from the training targets.

    Parameters
    ----------
    quantiles : sequence of float, optional (default: [0.1, 0.25, 0.5, 0.75, 0.9])
        Quantiles to predict in (0,1). They will be sorted ascending internally.
    group_col : str or None, optional (default: None)
        If provided, compute/store quantiles per group using X[group_col] at fit time.
        During predict, each row uses its group's quantiles; unseen groups fall back to
        global quantiles.
    """

    def __init__(
        self,
        quantiles: Sequence[float] | None = None,
        group_col: str | None = None,
    ):
        self.quantiles = list(quantiles) if quantiles else [0.1, 0.25, 0.5, 0.75, 0.9]
        self.group_col = group_col

        # Fitted attributes
        self.quantiles_: list[float] | None = None
        self.global_quantiles_: np.ndarray | None = None  # shape (n_q,)
        self.group_quantiles_: dict[Hashable, np.ndarray] | None = None  # each (n_q,)
        self.n_features_in_: int | None = None
        self.feature_names_in_: list[str] | None = None
        self._is_fitted: bool = False

    # ---- sklearn API ----
    def fit(
        self,
        X: Frame | Array,
        y: Series | Array,
        **fit_params: Any,
    ) -> HistoricalMultiQuantileBaseline:
        """
        Fit by computing empirical quantiles of y (globally or per group).
        """
        _ = fit_params  # unused, kept for compatibility with other estimators

        y_arr = self._to_1d_numpy(y)
        q_sorted = np.sort(np.asarray(self.quantiles, dtype=float))
        self._validate_quantiles(q_sorted)

        # Basic metadata (not strictly needed but nice for consistency)
        if hasattr(X, "shape"):
            self.n_features_in_ = X.shape[1] if len(np.shape(X)) == 2 else None
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)

        # Global quantiles
        self.global_quantiles_ = np.quantile(y_arr, q_sorted)

        # Optional per-group quantiles
        if self.group_col is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    f"group_col='{self.group_col}' requires X to be a pandas DataFrame."
                )
            if self.group_col not in X.columns:
                raise ValueError(
                    f"group_col '{self.group_col}' not found in X columns."
                )

            df = pd.DataFrame(
                {self.group_col: X[self.group_col].to_numpy(), "__y__": y_arr}
            )
            self.group_quantiles_ = {}
            for g, sub in df.groupby(self.group_col, sort=False, observed=True):
                self.group_quantiles_[g] = np.quantile(
                    sub["__y__"].to_numpy(), q_sorted
                )
        else:
            self.group_quantiles_ = None

        self.quantiles_ = q_sorted.tolist()
        self._is_fitted = True
        return self

    def predict(self, X: Frame | Array, return_full_quantiles: bool = False) -> Array:
        """
        Predict quantiles by repeating the (global or per-group) fitted quantiles.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
        return_full_quantiles : bool, default=False
            If True, returns array of shape (n_samples, n_quantiles).
            If False, returns the median quantile (n_samples,).

        Returns
        -------
        preds : ndarray
            Median predictions or full quantile matrix.
        """
        self._check_is_fitted()

        n_samples = self._num_rows(X)
        q_vec = np.asarray(self.quantiles_, dtype=float)
        n_q = q_vec.size

        if self.group_col is None:
            Q = np.tile(self.global_quantiles_[None, :], (n_samples, 1))
        else:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    f"group_col='{self.group_col}' requires X to be a pandas DataFrame."
                )
            if self.group_col not in X.columns:
                raise ValueError(
                    f"group_col '{self.group_col}' not found in X columns."
                )
            groups = X[self.group_col].to_numpy()
            Q = np.empty((n_samples, n_q), dtype=float)
            for i, g in enumerate(groups):
                if g in self.group_quantiles_:
                    Q[i, :] = self.group_quantiles_[g]
                else:
                    # Fallback to global if unseen group at inference time
                    Q[i, :] = self.global_quantiles_

        # Ensure non-crossing (should already be sorted, but just in case)
        Q.sort(axis=1)

        if return_full_quantiles:
            return Q

        median_idx = n_q // 2
        return Q[:, median_idx]

    def transform(self, X: Frame | Array) -> Array:
        """Alias for predict to enable transformer usage in pipelines."""
        return self.predict(X)

    def score(
        self, X: Frame | Array, y: Series | Array, sample_weight: Any = None
    ) -> float:
        """R² of median quantile (same behavior as your CatBoost wrapper)."""
        y_arr = self._to_1d_numpy(y)
        y_hat = self.predict(X)
        return r2_score(y_arr, y_hat, sample_weight=sample_weight)

    # ---- sklearn plumbing ----
    def get_params(self) -> dict[str, Any]:
        return {"quantiles": self.quantiles, "group_col": self.group_col}

    def set_params(self, **params: Any) -> HistoricalMultiQuantileBaseline:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ---- helpers ----
    @staticmethod
    def _to_1d_numpy(y: Series | Array) -> np.ndarray:
        if isinstance(y, pd.Series | pd.DataFrame):
            y = np.asarray(y).ravel()
        else:
            y = np.asarray(y).ravel()
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        return y

    @staticmethod
    def _validate_quantiles(q: np.ndarray) -> None:
        if q.size == 0:
            raise ValueError("quantiles must be non-empty.")
        if np.any((q <= 0) | (q >= 1)):
            raise ValueError("quantiles must be in the open interval (0, 1).")
        if len(np.unique(q)) != len(q):
            raise ValueError("quantiles must be unique.")

    @staticmethod
    def _num_rows(X: Frame | Array) -> int:
        if hasattr(X, "shape"):
            return int(X.shape[0])
        # Fall back to len()
        return len(X)

    def _check_is_fitted(self) -> None:
        if not self._is_fitted or self.global_quantiles_ is None:
            raise ValueError("Estimator is not fitted yet. Call 'fit' first.")

    # Optional for parity with your CatBoost wrapper
    @property
    def feature_importances_(self):
        raise AttributeError(
            "HistoricalMultiQuantileBaseline has no feature_importances_."
        )

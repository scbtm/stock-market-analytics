"""Naive baseline quantile regressors for comparison with advanced models."""

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y

from .protocols import SupportsPredictQuantiles

NDArrayF = npt.NDArray[np.float64]


class HistoricalQuantileBaseline(BaseEstimator, RegressorMixin, SupportsPredictQuantiles):
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

    def predict_quantiles(
        self,
        X: Any,
        quantiles: Sequence[float] | NDArrayF | None = None
    ) -> NDArrayF:
        """
        Predict quantiles using historical empirical quantiles.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features (ignored)
        quantiles : Sequence[float] | NDArrayF | None
            Quantiles to predict. If None, uses model's internal quantiles.
            
        Returns
        -------
        predictions : NDArrayF of shape (n_samples, n_quantiles)
            Quantile predictions (same for all samples)
        """
        if quantiles is not None:
            # Check if requested quantiles match model's quantiles
            requested = np.asarray(quantiles, dtype=float)
            model_quantiles = np.asarray(self.quantiles, dtype=float)
            
            if not np.array_equal(requested, model_quantiles):
                raise ValueError(
                    f"Requested quantiles {requested.tolist()} do not match "
                    f"model's trained quantiles {model_quantiles.tolist()}. "
                    "Historical baseline cannot predict arbitrary quantiles."
                )
        
        # Use regular predict method which returns quantiles
        return self.predict(X)

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return R^2 score for median quantile prediction."""
        from sklearn.metrics import r2_score

        predictions = self.predict(X)
        median_idx = len(self.quantiles) // 2
        median_pred = predictions[:, median_idx]

        return r2_score(y, median_pred, sample_weight=sample_weight)

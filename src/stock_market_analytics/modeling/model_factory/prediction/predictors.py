"""
Scikit-learn compatible model implementations.

Clean, focused implementations that follow our simplified protocol.
"""

from typing import Any, Sequence
import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y

from ..protocols import QuantilePredictor
from .prediction_functions import create_catboost_pool, predict_quantiles_catboost


class CatBoostMultiQuantileModel(BaseEstimator, RegressorMixin, QuantilePredictor):
    """Scikit-learn compatible wrapper for CatBoost multi-quantile regression."""

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
        self._model = None

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "CatBoostMultiQuantileModel":
        """Fit the CatBoost multi-quantile model."""
        X, y = check_X_y(X, y, accept_sparse=False)

        # Build CatBoost parameters
        params = self.catboost_params.copy()
        
        # Set up multi-quantile loss
        alpha_str = ",".join([str(q) for q in sorted(self.quantiles)])
        params["loss_function"] = f"MultiQuantile:alpha={alpha_str}"
        params["random_state"] = self.random_state
        params["verbose"] = self.verbose

        # Create and fit model
        self._model = CatBoostRegressor(**params)
        train_pool = create_catboost_pool(X, y)
        self._model.fit(train_pool, **fit_params)

        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()

        return self

    def predict_quantiles(self, X: Any, quantiles: Sequence[float]) -> np.ndarray:
        """Generate multi-quantile predictions."""
        if self._model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Note: CatBoost MultiQuantile ignores the quantiles parameter 
        # and returns predictions for all trained quantiles
        return predict_quantiles_catboost(self._model, X)

    def predict(self, X: Any) -> np.ndarray:
        """Generate point predictions (median quantile)."""
        quantile_preds = self.predict_quantiles(X, [0.5])
        # Find median quantile (0.5)
        median_idx = None
        for i, q in enumerate(self.quantiles):
            if abs(q - 0.5) < 1e-6:  # Close to 0.5
                median_idx = i
                break
        
        if median_idx is None:
            # If no exact 0.5 quantile, use middle one
            median_idx = len(self.quantiles) // 2
            
        return quantile_preds[:, median_idx]
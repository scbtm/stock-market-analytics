from typing import Literal, Any
import numpy as np
import pandas as pd
from catboost import Pool

from stock_market_analytics.modeling.modeling_config import modeling_config

QUANTILES = modeling_config["QUANTILES"]
TARGET = modeling_config["TARGET"]


class BaselinePredictor:
    """
    Baseline predictor that mimics CatBoost interface for multiquantile regression.
    Provides various naive prediction strategies for comparison with trained models.
    """
    
    def __init__(
        self, 
        strategy: Literal["historical_quantiles", "random_walk", "random_noise", "seasonal_naive"] = "historical_quantiles",
        random_state: int = 42
    ):
        """
        Initialize baseline predictor.
        
        Args:
            strategy: Baseline strategy to use
            random_state: Random seed for reproducible results
        """
        self.strategy = strategy
        self.random_state = random_state
        self.quantiles = np.array(QUANTILES)
        self.n_quantiles = len(self.quantiles)
        
        # Will be set during fit
        self.historical_quantiles_ = None
        self.seasonal_patterns_ = None
        self.noise_scale_ = None
        self.is_fitted_ = False
        
    def fit(self, train_pool: Pool, **kwargs: Any) -> "BaselinePredictor":
        """
        Fit the baseline predictor using training data.
        
        Args:
            train_pool: CatBoost Pool with training data
            **kwargs: Ignored for compatibility with CatBoost interface
            
        Returns:
            self: Fitted predictor
        """
        y_train = train_pool.get_label()
        
        # Historical quantiles strategy
        self.historical_quantiles_ = np.quantile(y_train, self.quantiles)
        
        # For seasonal naive, we need the original dataframe with dates
        if self.strategy == "seasonal_naive":
            # Try to extract seasonal patterns if we have access to the data
            # This is a simplified version - in practice, you might pass more context
            self.seasonal_patterns_ = {
                'overall_mean': np.mean(y_train),
                'overall_std': np.std(y_train)
            }
        
        # Estimate noise scale for random noise strategy
        if self.strategy == "random_noise":
            self.noise_scale_ = np.std(y_train) * 0.1  # 10% of training volatility
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame | Pool) -> np.ndarray:
        """
        Generate baseline predictions.
        
        Args:
            X: Input features (DataFrame or Pool)
            
        Returns:
            Predictions array of shape (n_samples, n_quantiles)
        """
        if not self.is_fitted_:
            raise ValueError("Predictor must be fitted before making predictions")
            
        if isinstance(X, Pool):
            n_samples = X.num_row()
        else:
            n_samples = len(X)
            
        np.random.seed(self.random_state)
        
        if self.strategy == "historical_quantiles":
            return self._predict_historical_quantiles(n_samples)
        elif self.strategy == "random_walk":
            return self._predict_random_walk(n_samples)
        elif self.strategy == "random_noise":
            return self._predict_random_noise(n_samples)
        elif self.strategy == "seasonal_naive":
            return self._predict_seasonal_naive(n_samples)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _predict_historical_quantiles(self, n_samples: int) -> np.ndarray:
        """Use historical quantiles from training data as constant predictions."""
        return np.tile(self.historical_quantiles_, (n_samples, 1))
    
    def _predict_random_walk(self, n_samples: int) -> np.ndarray:
        """
        Assume random walk: tomorrow's return distribution centered at 0.
        Use training data quantiles but centered at zero.
        """
        # Center historical quantiles at 0 (random walk assumption)
        centered_quantiles = self.historical_quantiles_ - np.median(self.historical_quantiles_)
        return np.tile(centered_quantiles, (n_samples, 1))
    
    def _predict_random_noise(self, n_samples: int) -> np.ndarray:
        """
        Add random noise to historical quantiles to simulate uncertainty.
        """
        base_predictions = self.historical_quantiles_
        
        # Add small random perturbations
        noise = np.random.normal(0, self.noise_scale_, (n_samples, self.n_quantiles))
        predictions = np.tile(base_predictions, (n_samples, 1)) + noise
        
        # Ensure quantiles remain monotonic
        predictions.sort(axis=1)
        return predictions
    
    def _predict_seasonal_naive(self, n_samples: int) -> np.ndarray:
        """
        Use seasonal patterns with some variation.
        Simplified version that uses overall statistics.
        """
        # Use overall mean with some seasonal variation
        base_mean = self.seasonal_patterns_['overall_mean']
        base_std = self.seasonal_patterns_['overall_std']
        
        # Create seasonal variation (simplified)
        seasonal_means = np.random.normal(base_mean, base_std * 0.1, n_samples)
        
        # Convert to quantiles around each seasonal mean
        predictions = np.zeros((n_samples, self.n_quantiles))
        for i in range(n_samples):
            # Create quantiles around the seasonal mean
            centered_quantiles = self.historical_quantiles_ - np.median(self.historical_quantiles_)
            predictions[i] = seasonal_means[i] + centered_quantiles
            
        return predictions
    
    @property
    def best_iteration_(self) -> int:
        """Mock property for compatibility with CatBoost interface."""
        return 1
        
    def get_param_names(self) -> list[str]:
        """Mock method for compatibility."""
        return ["strategy", "random_state"]
    
    def get_params(self) -> dict:
        """Get predictor parameters."""
        return {
            "strategy": self.strategy,
            "random_state": self.random_state
        }
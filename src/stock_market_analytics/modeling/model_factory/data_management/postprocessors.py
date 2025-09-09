"""
Post-processors for applying business rules to model predictions.

This module contains classes that apply domain-specific business rules
and constraints to raw model predictions.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from stock_market_analytics.modeling.model_factory.protocols import PostProcessor


class ReturnConstraintProcessor(BaseEstimator, TransformerMixin):
    """
    Post-processor that applies realistic constraints to return predictions.
    
    Ensures predicted returns fall within reasonable bounds based on
    historical market behavior and volatility estimates.
    """
    
    def __init__(
        self,
        max_daily_return: float = 0.20,
        min_daily_return: float = -0.20,
        volatility_multiplier: float = 3.0
    ):
        """
        Initialize the return constraint processor.
        
        Args:
            max_daily_return: Maximum allowed daily return
            min_daily_return: Minimum allowed daily return
            volatility_multiplier: Multiplier for volatility-based bounds
        """
        self.max_daily_return = max_daily_return
        self.min_daily_return = min_daily_return
        self.volatility_multiplier = volatility_multiplier
        self.historical_volatility_: float | None = None
        
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ReturnConstraintProcessor":
        """
        Fit the processor by learning historical volatility.
        
        Args:
            X: Historical returns for volatility estimation
            y: Not used
            
        Returns:
            Self for method chaining
        """
        if X.ndim == 1:
            returns = X
        else:
            # Assume first column contains returns
            returns = X[:, 0]
            
        self.historical_volatility_ = float(np.std(returns))
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply constraints to predictions.
        
        Args:
            X: Raw predictions to constrain
            
        Returns:
            Constrained predictions
        """
        predictions = X.copy()
        
        # Apply hard bounds
        predictions = np.clip(predictions, self.min_daily_return, self.max_daily_return)
        
        # Apply volatility-based bounds if available
        if self.historical_volatility_ is not None:
            vol_bound = self.volatility_multiplier * self.historical_volatility_
            predictions = np.clip(predictions, -vol_bound, vol_bound)
            
        return predictions
    
    def apply_rules(self, predictions: np.ndarray, context: dict[str, Any]) -> np.ndarray:
        """
        Apply business rules to predictions.
        
        Args:
            predictions: Raw model predictions
            context: Additional context (e.g., 'volatility_estimate')
            
        Returns:
            Post-processed predictions
        """
        constrained = predictions.copy()
        
        # Use context volatility if available
        if 'volatility_estimate' in context:
            vol_estimate = context['volatility_estimate']
            if isinstance(vol_estimate, (int, float)):
                vol_bound = self.volatility_multiplier * vol_estimate
                constrained = np.clip(constrained, -vol_bound, vol_bound)
            elif hasattr(vol_estimate, '__len__'):
                # Array of volatility estimates
                vol_bounds = self.volatility_multiplier * np.array(vol_estimate)
                constrained = np.maximum(constrained, -vol_bounds)
                constrained = np.minimum(constrained, vol_bounds)
        
        # Apply hard bounds
        constrained = np.clip(constrained, self.min_daily_return, self.max_daily_return)
        
        return constrained
    
    def validate_predictions(self, predictions: np.ndarray) -> bool:
        """
        Validate that predictions meet constraints.
        
        Args:
            predictions: Predictions to validate
            
        Returns:
            True if all predictions are within bounds
        """
        within_hard_bounds = np.all(
            (predictions >= self.min_daily_return) & 
            (predictions <= self.max_daily_return)
        )
        
        if self.historical_volatility_ is not None:
            vol_bound = self.volatility_multiplier * self.historical_volatility_
            within_vol_bounds = np.all(
                (predictions >= -vol_bound) & 
                (predictions <= vol_bound)
            )
            return within_hard_bounds and within_vol_bounds
        
        return within_hard_bounds


class QuantileConsistencyProcessor(BaseEstimator, TransformerMixin):
    """
    Post-processor that ensures quantile predictions are monotonic.
    
    Enforces that predicted quantiles maintain proper ordering
    (lower quantiles <= higher quantiles).
    """
    
    def __init__(self, quantiles: list[float]):
        """
        Initialize the quantile consistency processor.
        
        Args:
            quantiles: List of quantile levels in ascending order
        """
        self.quantiles = sorted(quantiles)
        
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "QuantileConsistencyProcessor":
        """
        Fit method for sklearn compatibility.
        
        Args:
            X: Training data (not used)
            y: Training targets (not used)
            
        Returns:
            Self for method chaining
        """
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Enforce quantile consistency.
        
        Args:
            X: Quantile predictions with shape (n_samples, n_quantiles)
            
        Returns:
            Monotonic quantile predictions
        """
        if X.ndim != 2 or X.shape[1] != len(self.quantiles):
            raise ValueError(
                f"Expected shape (n_samples, {len(self.quantiles)}), got {X.shape}"
            )
        
        # Sort each row to ensure monotonicity
        sorted_predictions = np.sort(X, axis=1)
        
        return sorted_predictions
    
    def apply_rules(self, predictions: np.ndarray, context: dict[str, Any]) -> np.ndarray:
        """
        Apply quantile consistency rules.
        
        Args:
            predictions: Raw quantile predictions
            context: Additional context (not used)
            
        Returns:
            Consistent quantile predictions
        """
        return self.transform(predictions)
    
    def validate_predictions(self, predictions: np.ndarray) -> bool:
        """
        Validate quantile consistency.
        
        Args:
            predictions: Quantile predictions to validate
            
        Returns:
            True if quantiles are monotonic
        """
        if predictions.ndim != 2:
            return False
        
        # Check if each row is sorted
        for row in predictions:
            if not np.all(row[:-1] <= row[1:]):
                return False
        
        return True


class OutlierClippingProcessor(BaseEstimator, TransformerMixin):
    """
    Post-processor that clips extreme outliers in predictions.
    
    Uses statistical methods to identify and clip predictions that are
    unreasonably extreme compared to historical patterns.
    """
    
    def __init__(
        self,
        method: str = "iqr",
        iqr_multiplier: float = 1.5,
        z_threshold: float = 3.0
    ):
        """
        Initialize the outlier clipping processor.
        
        Args:
            method: Outlier detection method ("iqr" or "zscore")
            iqr_multiplier: Multiplier for IQR-based outlier detection
            z_threshold: Z-score threshold for outlier detection
        """
        self.method = method
        self.iqr_multiplier = iqr_multiplier
        self.z_threshold = z_threshold
        self.lower_bound_: float | None = None
        self.upper_bound_: float | None = None
        self.mean_: float | None = None
        self.std_: float | None = None
        
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "OutlierClippingProcessor":
        """
        Fit the processor by learning outlier bounds.
        
        Args:
            X: Historical data for bound estimation
            y: Not used
            
        Returns:
            Self for method chaining
        """
        if X.ndim == 1:
            data = X
        else:
            # Assume first column contains the target variable
            data = X[:, 0]
        
        if self.method == "iqr":
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            self.lower_bound_ = q1 - self.iqr_multiplier * iqr
            self.upper_bound_ = q3 + self.iqr_multiplier * iqr
        elif self.method == "zscore":
            self.mean_ = float(np.mean(data))
            self.std_ = float(np.std(data))
            self.lower_bound_ = self.mean_ - self.z_threshold * self.std_
            self.upper_bound_ = self.mean_ + self.z_threshold * self.std_
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Clip outliers in predictions.
        
        Args:
            X: Predictions to clip
            
        Returns:
            Clipped predictions
        """
        if self.lower_bound_ is None or self.upper_bound_ is None:
            raise ValueError("Processor must be fitted before transforming")
        
        return np.clip(X, self.lower_bound_, self.upper_bound_)
    
    def apply_rules(self, predictions: np.ndarray, context: dict[str, Any]) -> np.ndarray:
        """
        Apply outlier clipping rules.
        
        Args:
            predictions: Raw predictions
            context: Additional context (not used)
            
        Returns:
            Clipped predictions
        """
        return self.transform(predictions)
    
    def validate_predictions(self, predictions: np.ndarray) -> bool:
        """
        Validate that predictions are within bounds.
        
        Args:
            predictions: Predictions to validate
            
        Returns:
            True if all predictions are within bounds
        """
        if self.lower_bound_ is None or self.upper_bound_ is None:
            return False
        
        return np.all(
            (predictions >= self.lower_bound_) & 
            (predictions <= self.upper_bound_)
        )


class MarketRegimeProcessor(BaseEstimator, TransformerMixin):
    """
    Post-processor that adjusts predictions based on market regime.
    
    Applies different processing rules depending on market conditions
    such as high volatility periods, bull/bear markets, etc.
    """
    
    def __init__(
        self,
        volatility_threshold: float = 0.02,
        bear_market_adjustment: float = 0.8,
        high_vol_adjustment: float = 0.9
    ):
        """
        Initialize the market regime processor.
        
        Args:
            volatility_threshold: Threshold for high volatility regime
            bear_market_adjustment: Multiplier for bear market predictions
            high_vol_adjustment: Multiplier for high volatility predictions
        """
        self.volatility_threshold = volatility_threshold
        self.bear_market_adjustment = bear_market_adjustment
        self.high_vol_adjustment = high_vol_adjustment
        
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MarketRegimeProcessor":
        """
        Fit method for sklearn compatibility.
        
        Args:
            X: Training data (not used)
            y: Training targets (not used)
            
        Returns:
            Self for method chaining
        """
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform predictions (identity transform without context).
        
        Args:
            X: Predictions
            
        Returns:
            Unchanged predictions (use apply_rules for regime adjustments)
        """
        return X
    
    def apply_rules(self, predictions: np.ndarray, context: dict[str, Any]) -> np.ndarray:
        """
        Apply market regime adjustments.
        
        Args:
            predictions: Raw predictions
            context: Market context containing 'volatility', 'market_trend', etc.
            
        Returns:
            Regime-adjusted predictions
        """
        adjusted_predictions = predictions.copy()
        
        # High volatility adjustment
        if 'volatility' in context:
            volatility = context['volatility']
            if isinstance(volatility, (int, float)):
                if volatility > self.volatility_threshold:
                    adjusted_predictions *= self.high_vol_adjustment
            elif hasattr(volatility, '__len__'):
                # Array of volatilities
                vol_array = np.array(volatility)
                high_vol_mask = vol_array > self.volatility_threshold
                adjusted_predictions[high_vol_mask] *= self.high_vol_adjustment
        
        # Bear market adjustment
        if 'market_trend' in context:
            trend = context['market_trend']
            if trend == 'bear':
                # Reduce positive predictions in bear markets
                positive_mask = adjusted_predictions > 0
                adjusted_predictions[positive_mask] *= self.bear_market_adjustment
            elif hasattr(trend, '__len__'):
                # Array of market trends
                bear_mask = np.array(trend) == 'bear'
                positive_mask = adjusted_predictions > 0
                bear_and_positive = bear_mask & positive_mask
                adjusted_predictions[bear_and_positive] *= self.bear_market_adjustment
        
        return adjusted_predictions
    
    def validate_predictions(self, predictions: np.ndarray) -> bool:
        """
        Validate predictions (always returns True as no hard constraints).
        
        Args:
            predictions: Predictions to validate
            
        Returns:
            Always True
        """
        return True


class CompositePredictionProcessor(BaseEstimator, TransformerMixin):
    """
    Composite post-processor that applies multiple processing steps in sequence.
    
    Combines multiple post-processors for comprehensive prediction refinement.
    """
    
    def __init__(self, processors: list[PostProcessor]):
        """
        Initialize the composite processor.
        
        Args:
            processors: List of post-processors to apply in sequence
        """
        self.processors = processors
        
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "CompositePredictionProcessor":
        """
        Fit all processors.
        
        Args:
            X: Training data
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        for processor in self.processors:
            if hasattr(processor, 'fit'):
                processor.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply all processors in sequence.
        
        Args:
            X: Predictions to process
            
        Returns:
            Processed predictions
        """
        result = X.copy()
        for processor in self.processors:
            if hasattr(processor, 'transform'):
                result = processor.transform(result)
        return result
    
    def apply_rules(self, predictions: np.ndarray, context: dict[str, Any]) -> np.ndarray:
        """
        Apply all processor rules in sequence.
        
        Args:
            predictions: Raw predictions
            context: Processing context
            
        Returns:
            Fully processed predictions
        """
        result = predictions.copy()
        for processor in self.processors:
            result = processor.apply_rules(result, context)
        return result
    
    def validate_predictions(self, predictions: np.ndarray) -> bool:
        """
        Validate predictions against all processors.
        
        Args:
            predictions: Predictions to validate
            
        Returns:
            True if all processors validate successfully
        """
        return all(
            processor.validate_predictions(predictions) 
            for processor in self.processors
        )
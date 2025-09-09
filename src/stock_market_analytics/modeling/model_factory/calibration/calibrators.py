"""
Calibrator implementations for post-hoc prediction processing.

This module contains calibrator classes that implement the calibration protocols
and can be integrated into sklearn pipelines.
"""

from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

from stock_market_analytics.modeling.model_factory.protocols import QuantileCalibrator, ProbabilityCalibrator
from stock_market_analytics.modeling.model_factory.calibration.calibration_functions import (
    conformity_scores,
    quantile_conformal_bounds,
    normalized_conformity_scores,
    isotonic_regression_calibration,
)


class QuantileConformalCalibrator(BaseEstimator, TransformerMixin):
    """
    Quantile conformal calibrator for prediction intervals.
    
    This calibrator learns conformity scores from a calibration set and 
    provides prediction intervals with guaranteed coverage.
    """
    
    def __init__(self, alpha: float = 0.1, method: str = "simple"):
        """
        Initialize the quantile conformal calibrator.
        
        Args:
            alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
            method: Conformity score method ("simple" or "normalized")
        """
        self.alpha = alpha
        self.method = method
        self.conformity_scores_: np.ndarray | None = None
        
    def calibrate(self, X_cal: pl.DataFrame, y_cal: pl.Series, **kwargs: Any) -> "QuantileConformalCalibrator":
        """Learn conformity scores from calibration data."""
        return self.fit(X_cal, y_cal, **kwargs)
    
    def fit(self, X_cal: pl.DataFrame, y_cal: pl.Series, y_pred_cal: np.ndarray | None = None) -> "QuantileConformalCalibrator":
        """
        Fit the conformal calibrator.
        
        Args:
            X_cal: Calibration features (not used in this implementation)
            y_cal: Calibration targets
            y_pred_cal: Point predictions on calibration set
            
        Returns:
            Self for method chaining
        """
        if y_pred_cal is None:
            raise ValueError("y_pred_cal must be provided for conformal calibration")
            
        y_true = y_cal.to_numpy()
        
        if self.method == "simple":
            self.conformity_scores_ = conformity_scores(y_true, y_pred_cal)
        elif self.method == "normalized":
            # For normalized method, we'd need predicted uncertainties
            # For now, fall back to simple method
            self.conformity_scores_ = conformity_scores(y_true, y_pred_cal)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return self
    
    def predict(self, y_hat: np.ndarray) -> np.ndarray:
        """
        Apply conformal calibration to point predictions.
        
        Args:
            y_hat: Point predictions
            
        Returns:
            Prediction intervals as (n_samples, 2) array with [lower, upper] bounds
        """
        if self.conformity_scores_ is None:
            raise ValueError("Calibrator must be fitted before making predictions")
            
        lower_bounds, upper_bounds = quantile_conformal_bounds(
            self.conformity_scores_, self.alpha, y_hat
        )
        
        return np.column_stack([lower_bounds, upper_bounds])
    
    def predict_quantiles(self, y_hat: np.ndarray, quantiles: list[float]) -> np.ndarray:
        """
        Predict quantiles based on conformal intervals.
        
        Args:
            y_hat: Point predictions
            quantiles: List of quantiles to predict
            
        Returns:
            Quantile predictions with shape (n_samples, n_quantiles)
        """
        if self.conformity_scores_ is None:
            raise ValueError("Calibrator must be fitted before making predictions")
            
        predictions = []
        for q in quantiles:
            alpha_q = 2 * min(q, 1 - q)  # Convert quantile to alpha
            lower_bounds, upper_bounds = quantile_conformal_bounds(
                self.conformity_scores_, alpha_q, y_hat
            )
            
            if q <= 0.5:
                predictions.append(lower_bounds)
            else:
                predictions.append(upper_bounds)
                
        return np.column_stack(predictions)


class AdaptiveConformalCalibrator(BaseEstimator, TransformerMixin):
    """
    Adaptive conformal calibrator for online prediction intervals.
    
    This calibrator adapts its threshold based on recent performance,
    suitable for non-stationary environments.
    """
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.005, window_size: int = 100):
        """
        Initialize the adaptive conformal calibrator.
        
        Args:
            alpha: Target miscoverage level
            gamma: Learning rate for adaptation
            window_size: Size of the sliding window for adaptation
        """
        self.alpha = alpha
        self.gamma = gamma
        self.window_size = window_size
        self.conformity_scores_: list[float] = []
        self.current_threshold_: float = 0.0
        
    def calibrate(self, X_cal: pl.DataFrame, y_cal: pl.Series, **kwargs: Any) -> "AdaptiveConformalCalibrator":
        """Learn initial conformity scores from calibration data."""
        return self.fit(X_cal, y_cal, **kwargs)
    
    def fit(self, X_cal: pl.DataFrame, y_cal: pl.Series, y_pred_cal: np.ndarray | None = None) -> "AdaptiveConformalCalibrator":
        """
        Fit the adaptive conformal calibrator.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
            y_pred_cal: Point predictions on calibration set
            
        Returns:
            Self for method chaining
        """
        if y_pred_cal is None:
            raise ValueError("y_pred_cal must be provided for conformal calibration")
            
        y_true = y_cal.to_numpy()
        initial_scores = conformity_scores(y_true, y_pred_cal)
        
        self.conformity_scores_ = initial_scores.tolist()
        self.current_threshold_ = float(np.quantile(initial_scores, 1 - self.alpha))
        
        return self
    
    def predict(self, y_hat: np.ndarray) -> np.ndarray:
        """
        Apply adaptive conformal calibration.
        
        Args:
            y_hat: Point predictions
            
        Returns:
            Prediction intervals as (n_samples, 2) array
        """
        if not self.conformity_scores_:
            raise ValueError("Calibrator must be fitted before making predictions")
            
        # Use current adaptive threshold
        lower_bounds = y_hat - self.current_threshold_
        upper_bounds = y_hat + self.current_threshold_
        
        return np.column_stack([lower_bounds, upper_bounds])
    
    def update(self, y_true: float, y_pred: float) -> None:
        """
        Update the calibrator with new observation.
        
        Args:
            y_true: True value
            y_pred: Predicted value
        """
        new_score = abs(y_true - y_pred)
        self.conformity_scores_.append(new_score)
        
        # Maintain sliding window
        if len(self.conformity_scores_) > self.window_size:
            self.conformity_scores_.pop(0)
        
        # Update threshold
        scores_array = np.array(self.conformity_scores_)
        new_threshold = np.quantile(scores_array, 1 - self.alpha)
        
        # Adaptive update
        self.current_threshold_ = (
            (1 - self.gamma) * self.current_threshold_ + 
            self.gamma * new_threshold
        )


class PlattScalingCalibrator(BaseEstimator, TransformerMixin):
    """
    Platt scaling calibrator for probability calibration.
    
    This calibrator fits a sigmoid function to map prediction scores to
    calibrated probabilities.
    """
    
    def __init__(self, max_iter: int = 1000):
        """
        Initialize the Platt scaling calibrator.
        
        Args:
            max_iter: Maximum iterations for sigmoid fitting
        """
        self.max_iter = max_iter
        self.sigmoid_a_: float = 0.0
        self.sigmoid_b_: float = 0.0
        
    def calibrate(self, X_cal: pl.DataFrame, y_cal: pl.Series, **kwargs: Any) -> "PlattScalingCalibrator":
        """Learn sigmoid parameters from calibration data."""
        return self.fit(X_cal, y_cal, **kwargs)
    
    def fit(self, X_cal: pl.DataFrame, y_cal: pl.Series, scores_cal: np.ndarray | None = None) -> "PlattScalingCalibrator":
        """
        Fit the Platt scaling calibrator.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration binary targets
            scores_cal: Prediction scores on calibration set
            
        Returns:
            Self for method chaining
        """
        if scores_cal is None:
            raise ValueError("scores_cal must be provided for Platt scaling")
            
        from sklearn.linear_model import LogisticRegression
        
        y_true = y_cal.to_numpy()
        
        # Fit logistic regression on scores
        lr = LogisticRegression(max_iter=self.max_iter)
        lr.fit(scores_cal.reshape(-1, 1), y_true)
        
        self.sigmoid_a_ = float(lr.coef_[0, 0])
        self.sigmoid_b_ = float(lr.intercept_[0])
        
        return self
    
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to prediction scores.
        
        Args:
            scores: Raw prediction scores
            
        Returns:
            Calibrated probabilities
        """
        if self.sigmoid_a_ == 0.0 and self.sigmoid_b_ == 0.0:
            raise ValueError("Calibrator must be fitted before making predictions")
            
        # Apply sigmoid function: P(y=1|score) = 1 / (1 + exp(a*score + b))
        linear_scores = self.sigmoid_a_ * scores + self.sigmoid_b_
        probabilities = 1.0 / (1.0 + np.exp(-linear_scores))
        
        return probabilities
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            scores: Raw prediction scores
            
        Returns:
            Class probabilities as (n_samples, 2) array
        """
        pos_probs = self.predict(scores)
        neg_probs = 1.0 - pos_probs
        
        return np.column_stack([neg_probs, pos_probs])


class IsotonicRegressionCalibrator(BaseEstimator, TransformerMixin):
    """
    Isotonic regression calibrator for probability calibration.
    
    This calibrator uses isotonic regression to map prediction scores to
    calibrated probabilities, providing more flexibility than Platt scaling.
    """
    
    def __init__(self):
        """Initialize the isotonic regression calibrator."""
        self.calibration_curve_: tuple[np.ndarray, np.ndarray] | None = None
        
    def calibrate(self, X_cal: pl.DataFrame, y_cal: pl.Series, **kwargs: Any) -> "IsotonicRegressionCalibrator":
        """Learn isotonic regression from calibration data."""
        return self.fit(X_cal, y_cal, **kwargs)
    
    def fit(self, X_cal: pl.DataFrame, y_cal: pl.Series, scores_cal: np.ndarray | None = None) -> "IsotonicRegressionCalibrator":
        """
        Fit the isotonic regression calibrator.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration binary targets
            scores_cal: Prediction scores on calibration set
            
        Returns:
            Self for method chaining
        """
        if scores_cal is None:
            raise ValueError("scores_cal must be provided for isotonic regression")
            
        y_true = y_cal.to_numpy()
        
        self.calibration_curve_ = isotonic_regression_calibration(scores_cal, y_true)
        
        return self
    
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply isotonic regression calibration to prediction scores.
        
        Args:
            scores: Raw prediction scores
            
        Returns:
            Calibrated probabilities
        """
        if self.calibration_curve_ is None:
            raise ValueError("Calibrator must be fitted before making predictions")
            
        sorted_scores, calibrated_probs = self.calibration_curve_
        
        # Interpolate to get calibrated probabilities for new scores
        calibrated = np.interp(scores, sorted_scores, calibrated_probs)
        
        return calibrated
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            scores: Raw prediction scores
            
        Returns:
            Class probabilities as (n_samples, 2) array
        """
        pos_probs = self.predict(scores)
        neg_probs = 1.0 - pos_probs
        
        return np.column_stack([neg_probs, pos_probs])
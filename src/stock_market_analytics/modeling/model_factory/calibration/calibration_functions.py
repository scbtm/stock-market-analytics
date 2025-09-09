"""
Helper functions for calibration methods.

This module provides mathematical functions and utilities needed to perform 
various calibration tasks on model predictions.
"""

import numpy as np
from typing import Tuple


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the pinball loss for quantile predictions.
    
    Args:
        y_true: True target values
        y_pred: Predicted quantile values
        quantile: The quantile level (e.g., 0.1 for 10th percentile)
        
    Returns:
        Mean pinball loss
    """
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)


def coverage_score(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calculate empirical coverage for prediction intervals.
    
    Args:
        y_true: True target values
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        
    Returns:
        Empirical coverage rate (proportion of observations within intervals)
    """
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    return float(coverage)


def interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> np.ndarray:
    """
    Calculate prediction interval widths.
    
    Args:
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        
    Returns:
        Array of interval widths
    """
    return y_upper - y_lower


def mean_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calculate mean prediction interval width.
    
    Args:
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        
    Returns:
        Mean interval width
    """
    return float(np.mean(interval_width(y_lower, y_upper)))


def conformity_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate conformity scores for conformal prediction.
    
    Args:
        y_true: True target values
        y_pred: Point predictions
        
    Returns:
        Array of conformity scores (absolute residuals)
    """
    return np.abs(y_true - y_pred)


def quantile_conformal_bounds(
    conformity_scores: np.ndarray, 
    alpha: float, 
    y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate conformal prediction bounds using quantile method.
    
    Args:
        conformity_scores: Array of conformity scores from calibration set
        alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        y_pred: Point predictions for test set
        
    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    n = len(conformity_scores)
    quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
    quantile_value = np.quantile(conformity_scores, quantile_level)
    
    lower_bounds = y_pred - quantile_value
    upper_bounds = y_pred + quantile_value
    
    return lower_bounds, upper_bounds


def normalized_conformity_scores(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_std: np.ndarray
) -> np.ndarray:
    """
    Calculate normalized conformity scores for conformal prediction.
    
    Args:
        y_true: True target values
        y_pred: Point predictions
        y_std: Predicted standard deviations or uncertainty estimates
        
    Returns:
        Array of normalized conformity scores
    """
    return np.abs(y_true - y_pred) / np.maximum(y_std, 1e-8)  # Avoid division by zero


def conditional_coverage_by_group(
    y_true: np.ndarray, 
    y_lower: np.ndarray, 
    y_upper: np.ndarray, 
    groups: np.ndarray
) -> dict[str, float]:
    """
    Calculate conditional coverage for different groups.
    
    Args:
        y_true: True target values
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        groups: Group identifiers for each sample
        
    Returns:
        Dictionary mapping group labels to coverage rates
    """
    coverage_by_group = {}
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        mask = groups == group
        if np.sum(mask) > 0:
            group_coverage = coverage_score(
                y_true[mask], y_lower[mask], y_upper[mask]
            )
            coverage_by_group[str(group)] = group_coverage
    
    return coverage_by_group


def adaptive_conformal_threshold(
    residuals: np.ndarray, 
    alpha: float, 
    gamma: float = 0.005
) -> float:
    """
    Calculate adaptive threshold for online conformal prediction.
    
    Args:
        residuals: Historical conformity scores
        alpha: Target miscoverage level
        gamma: Learning rate for adaptation
        
    Returns:
        Adaptive threshold value
    """
    n = len(residuals)
    if n == 0:
        return 0.0
    
    # Adaptive threshold based on recent performance
    recent_weight = min(1.0, gamma * n)
    base_quantile = np.quantile(residuals, 1 - alpha)
    
    # Adjust based on recent miscoverage
    recent_residuals = residuals[-max(1, int(0.1 * n)):]
    recent_quantile = np.quantile(recent_residuals, 1 - alpha)
    
    threshold = (1 - recent_weight) * base_quantile + recent_weight * recent_quantile
    return float(threshold)


def isotonic_regression_calibration(
    y_scores: np.ndarray, 
    y_true: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform isotonic regression for probability calibration.
    
    Args:
        y_scores: Uncalibrated prediction scores
        y_true: True binary labels
        
    Returns:
        Tuple of (sorted_scores, calibrated_probabilities)
    """
    from sklearn.isotonic import IsotonicRegression
    
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    sorted_indices = np.argsort(y_scores)
    sorted_scores = y_scores[sorted_indices]
    sorted_labels = y_true[sorted_indices]
    
    calibrated_probs = iso_reg.fit_transform(sorted_scores, sorted_labels)
    
    return sorted_scores, calibrated_probs
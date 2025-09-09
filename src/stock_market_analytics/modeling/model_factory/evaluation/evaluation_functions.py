"""
Helper functions for model evaluation.

This module provides mathematical functions and utilities needed to perform
various evaluation tasks for machine learning models in financial contexts.
"""

import numpy as np
from typing import Tuple, Dict


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = y_true != 0
    if not np.any(mask):
        return float('inf')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (proportion of correct direction predictions).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Proportion of predictions with correct direction
    """
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    return float(np.mean(true_direction == pred_direction))


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculate hit rate (proportion of predictions above threshold that are correct).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        threshold: Threshold for positive predictions
        
    Returns:
        Hit rate for predictions above threshold
    """
    positive_predictions = y_pred > threshold
    if not np.any(positive_predictions):
        return 0.0
    
    correct_positives = (y_true > threshold) & positive_predictions
    return float(np.sum(correct_positives) / np.sum(positive_predictions))


def sharpe_ratio_proxy(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio proxy for predicted returns.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sharpe ratio proxy
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    if np.std(excess_returns) == 0:
        return 0.0
    return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio for predicted returns.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.std(downside_returns)
    return float(np.mean(excess_returns) / downside_deviation * np.sqrt(252))


def maximum_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from cumulative returns.
    
    Args:
        cumulative_returns: Cumulative returns series
        
    Returns:
        Maximum drawdown (positive value)
    """
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return float(-np.min(drawdown))


def calmar_ratio(returns: np.ndarray) -> float:
    """
    Calculate Calmar ratio (annualized return / maximum drawdown).
    
    Args:
        returns: Array of returns
        
    Returns:
        Calmar ratio
    """
    cumulative_returns = np.cumprod(1 + returns)
    max_dd = maximum_drawdown(cumulative_returns)
    
    if max_dd == 0:
        return float('inf') if np.mean(returns) > 0 else 0.0
    
    annualized_return = (cumulative_returns[-1] ** (252 / len(returns))) - 1
    return float(annualized_return / max_dd)


def information_ratio(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate information ratio.
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Information ratio
    """
    active_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(active_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return float(np.mean(active_returns) / tracking_error)


def value_at_risk(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        VaR value (positive for losses)
    """
    return float(-np.percentile(returns, confidence_level * 100))


def expected_shortfall(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level
        
    Returns:
        Expected shortfall (positive for losses)
    """
    var = value_at_risk(returns, confidence_level)
    tail_losses = returns[returns <= -var]
    
    if len(tail_losses) == 0:
        return var
    
    return float(-np.mean(tail_losses))


def pinball_loss_vectorized(y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """
    Vectorized pinball loss calculation for multiple quantiles.
    
    Args:
        y_true: True values (n_samples,)
        y_pred: Predicted quantiles (n_samples, n_quantiles)
        quantiles: Quantile levels (n_quantiles,)
        
    Returns:
        Pinball losses for each quantile (n_quantiles,)
    """
    errors = y_true[:, np.newaxis] - y_pred  # Shape: (n_samples, n_quantiles)
    losses = np.where(
        errors >= 0,
        quantiles[np.newaxis, :] * errors,
        (quantiles[np.newaxis, :] - 1) * errors
    )
    return np.mean(losses, axis=0)


def quantile_coverage(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """
    Calculate empirical coverage for quantile predictions.
    
    Args:
        y_true: True values
        y_pred_quantiles: Predicted quantiles (n_samples, n_quantiles)
        quantiles: Quantile levels
        
    Returns:
        Empirical coverage for each quantile
    """
    coverage = np.mean(y_true[:, np.newaxis] <= y_pred_quantiles, axis=0)
    return coverage


def quantile_loss_differential(
    y_true: np.ndarray, 
    y_pred_quantiles: np.ndarray, 
    quantiles: np.ndarray
) -> Dict[str, float]:
    """
    Calculate differential quantile loss metrics.
    
    Args:
        y_true: True values
        y_pred_quantiles: Predicted quantiles
        quantiles: Quantile levels
        
    Returns:
        Dictionary of differential loss metrics
    """
    pinball_losses = pinball_loss_vectorized(y_true, y_pred_quantiles, quantiles)
    coverage = quantile_coverage(y_true, y_pred_quantiles, quantiles)
    
    # Coverage deviations
    coverage_errors = coverage - quantiles
    
    return {
        'mean_pinball_loss': float(np.mean(pinball_losses)),
        'max_pinball_loss': float(np.max(pinball_losses)),
        'mean_coverage_error': float(np.mean(np.abs(coverage_errors))),
        'max_coverage_error': float(np.max(np.abs(coverage_errors))),
        'coverage_bias': float(np.mean(coverage_errors))
    }


def interval_score(
    y_true: np.ndarray, 
    y_lower: np.ndarray, 
    y_upper: np.ndarray, 
    alpha: float = 0.1
) -> float:
    """
    Calculate interval score for prediction intervals.
    
    Args:
        y_true: True values
        y_lower: Lower bounds of prediction intervals
        y_upper: Upper bounds of prediction intervals
        alpha: Miscoverage level (e.g., 0.1 for 90% intervals)
        
    Returns:
        Interval score (lower is better)
    """
    width = y_upper - y_lower
    
    # Penalties for being outside the interval
    lower_penalty = (2 / alpha) * (y_lower - y_true) * (y_true < y_lower)
    upper_penalty = (2 / alpha) * (y_true - y_upper) * (y_true > y_upper)
    
    total_score = width + lower_penalty + upper_penalty
    return float(np.mean(total_score))


def prediction_interval_coverage_probability(
    y_true: np.ndarray, 
    y_lower: np.ndarray, 
    y_upper: np.ndarray
) -> float:
    """
    Calculate empirical coverage probability for prediction intervals.
    
    Args:
        y_true: True values
        y_lower: Lower bounds
        y_upper: Upper bounds
        
    Returns:
        Coverage probability
    """
    coverage = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(coverage))


def mean_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calculate mean prediction interval width.
    
    Args:
        y_lower: Lower bounds
        y_upper: Upper bounds
        
    Returns:
        Mean interval width
    """
    return float(np.mean(y_upper - y_lower))


def normalized_interval_width(
    y_lower: np.ndarray, 
    y_upper: np.ndarray, 
    y_true: np.ndarray
) -> float:
    """
    Calculate normalized mean interval width.
    
    Args:
        y_lower: Lower bounds
        y_upper: Upper bounds
        y_true: True values for normalization
        
    Returns:
        Normalized mean interval width
    """
    width = y_upper - y_lower
    normalization = np.std(y_true)
    
    if normalization == 0:
        return float('inf')
    
    return float(np.mean(width) / normalization)
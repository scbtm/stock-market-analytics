"""
Helper functions for model evaluation.

This module provides mathematical functions and utilities needed to perform
various evaluation tasks for machine learning models in financial contexts.
"""

import numpy as np
from typing import Dict

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
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred)
    quantiles = np.asarray(quantiles).reshape(-1)

    if y_pred.ndim != 2:
        raise ValueError("y_pred must be 2D with shape (n_samples, n_quantiles).")
    n_samples, n_quantiles = y_pred.shape
    if y_true.shape[0] != n_samples:
        raise ValueError("y_true length must match y_pred.shape[0].")
    if quantiles.shape[0] != n_quantiles:
        raise ValueError("quantiles length must match y_pred.shape[1].")
    if np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError("All quantiles must be in [0, 1].")

    e = y_true[:, None] - y_pred                      # (n_samples, n_quantiles)
    q = quantiles[None, :]                            # (1, n_quantiles)
    # Pinball loss: max(q*e, (q-1)*e)
    losses = np.maximum(q * e, (q - 1.0) * e)
    return np.mean(losses, axis=0)


def quantile_coverage(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """
    Calculate empirical coverage for quantile predictions.

    Args:
        y_true: True values (n_samples,)
        y_pred_quantiles: Predicted quantiles (n_samples, n_quantiles)
        quantiles: Quantile levels (n_quantiles,)

    Returns:
        Empirical coverage for each quantile (n_quantiles,)
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_quantiles = np.asarray(y_pred_quantiles)
    quantiles = np.asarray(quantiles).reshape(-1)

    if y_pred_quantiles.ndim != 2:
        raise ValueError("y_pred_quantiles must be 2D with shape (n_samples, n_quantiles).")
    n_samples, n_quantiles = y_pred_quantiles.shape
    if y_true.shape[0] != n_samples:
        raise ValueError("y_true length must match y_pred_quantiles.shape[0].")
    if quantiles.shape[0] != n_quantiles:
        raise ValueError("quantiles length must match y_pred_quantiles.shape[1].")

    # P(Y <= Q_q) should be ≈ q
    return np.mean(y_true[:, None] <= y_pred_quantiles, axis=0)


def quantile_loss_differential(
    y_true: np.ndarray, 
    y_pred_quantiles: np.ndarray, 
    quantiles: np.ndarray
) -> dict[str, float]:
    """
    Calculate differential quantile loss metrics.

    Args:
        y_true: True values (n_samples,)
        y_pred_quantiles: Predicted quantiles (n_samples, n_quantiles)
        quantiles: Quantile levels (n_quantiles,)

    Returns:
        Dictionary of differential loss metrics
    """
    pinball_losses = pinball_loss_vectorized(y_true, y_pred_quantiles, quantiles)
    cov = quantile_coverage(y_true, y_pred_quantiles, quantiles)
    cov_errors = cov - quantiles

    return {
        'mean_pinball_loss': float(np.mean(pinball_losses)),
        'max_pinball_loss': float(np.max(pinball_losses)),
        'mean_coverage_error': float(np.mean(np.abs(cov_errors))),
        'max_coverage_error': float(np.max(np.abs(cov_errors))),
        'coverage_bias': float(np.mean(cov_errors)),  # <0 => under-coverage (preds too low)
    }


def interval_score(
    y_true: np.ndarray, 
    y_lower: np.ndarray, 
    y_upper: np.ndarray, 
    alpha: float = 0.1
) -> float:
    """
    Calculate interval score for prediction intervals (Gneiting & Raftery, 2007).

    Args:
        y_true: True values (n_samples,)
        y_lower: Lower bounds (n_samples,)
        y_upper: Upper bounds (n_samples,)
        alpha: Miscoverage level (e.g., 0.1 for 90% intervals)

    Returns:
        Interval score (lower is better)
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    y_true = np.asarray(y_true).reshape(-1)
    y_lower = np.asarray(y_lower).reshape(-1)
    y_upper = np.asarray(y_upper).reshape(-1)

    if not (y_true.shape == y_lower.shape == y_upper.shape):
        raise ValueError("y_true, y_lower, y_upper must have the same shape.")
    if np.any(y_lower > y_upper):
        raise ValueError("Found y_lower > y_upper; check for quantile crossing.")

    width = y_upper - y_lower
    below = y_true < y_lower
    above = y_true > y_upper

    lower_penalty = (2.0 / alpha) * (y_lower - y_true) * below
    upper_penalty = (2.0 / alpha) * (y_true - y_upper) * above

    total = width + lower_penalty + upper_penalty
    return float(np.mean(total))


def prediction_interval_coverage_probability(
    y_true: np.ndarray, 
    y_lower: np.ndarray, 
    y_upper: np.ndarray
) -> float:
    """
    Calculate empirical coverage probability for prediction intervals.

    Args:
        y_true: True values (n_samples,)
        y_lower: Lower bounds (n_samples,)
        y_upper: Upper bounds (n_samples,)

    Returns:
        Coverage probability in [0, 1]
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_lower = np.asarray(y_lower).reshape(-1)
    y_upper = np.asarray(y_upper).reshape(-1)

    if not (y_true.shape == y_lower.shape == y_upper.shape):
        raise ValueError("y_true, y_lower, y_upper must have the same shape.")
    if np.any(y_lower > y_upper):
        raise ValueError("Found y_lower > y_upper; check for quantile crossing.")

    covered = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(covered))


def mean_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calculate mean prediction interval width.

    Args:
        y_lower: Lower bounds (n_samples,)
        y_upper: Upper bounds (n_samples,)

    Returns:
        Mean interval width
    """
    y_lower = np.asarray(y_lower).reshape(-1)
    y_upper = np.asarray(y_upper).reshape(-1)
    if y_lower.shape != y_upper.shape:
        raise ValueError("y_lower and y_upper must have the same shape.")
    if np.any(y_lower > y_upper):
        raise ValueError("Found y_lower > y_upper; check for quantile crossing.")
    return float(np.mean(y_upper - y_lower))


def normalized_interval_width(
    y_lower: np.ndarray, 
    y_upper: np.ndarray, 
    y_true: np.ndarray
) -> float:
    """
    Calculate normalized mean interval width.

    Args:
        y_lower: Lower bounds (n_samples,)
        y_upper: Upper bounds (n_samples,)
        y_true: True values for normalization (n_samples,)

    Returns:
        Normalized mean interval width (mean width / std(y_true))
    """
    y_lower = np.asarray(y_lower).reshape(-1)
    y_upper = np.asarray(y_upper).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    if not (y_lower.shape == y_upper.shape == y_true.shape):
        raise ValueError("y_lower, y_upper, y_true must have the same shape.")
    if np.any(y_lower > y_upper):
        raise ValueError("Found y_lower > y_upper; check for quantile crossing.")

    width = y_upper - y_lower
    denom = float(np.std(y_true))
    if not np.isfinite(denom) or denom <= 0.0:
        # If std is 0, intervals with any positive width should flag as 'infinite' normalized width.
        return 0.0 if np.allclose(width, 0.0) else float('inf')

    return float(np.mean(width) / denom)



# def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """Calculate Mean Absolute Error."""
#     return float(np.mean(np.abs(y_true - y_pred)))


# def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """Calculate Mean Squared Error."""
#     return float(np.mean((y_true - y_pred) ** 2))


# def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """Calculate Root Mean Squared Error."""
#     return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """Calculate Mean Absolute Percentage Error."""
#     mask = y_true != 0
#     if not np.any(mask):
#         return float('inf')
#     return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """Calculate Symmetric Mean Absolute Percentage Error."""
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
#     mask = denominator != 0
#     if not np.any(mask):
#         return 0.0
#     return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


# def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """
#     Calculate directional accuracy (proportion of correct direction predictions).
    
#     Args:
#         y_true: True values
#         y_pred: Predicted values
        
#     Returns:
#         Proportion of predictions with correct direction
#     """
#     true_direction = np.sign(y_true)
#     pred_direction = np.sign(y_pred)
#     return float(np.mean(true_direction == pred_direction))


# def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0) -> float:
#     """
#     Calculate hit rate (proportion of predictions above threshold that are correct).
    
#     Args:
#         y_true: True values
#         y_pred: Predicted values
#         threshold: Threshold for positive predictions
        
#     Returns:
#         Hit rate for predictions above threshold
#     """
#     positive_predictions = y_pred > threshold
#     if not np.any(positive_predictions):
#         return 0.0
    
#     correct_positives = (y_true > threshold) & positive_predictions
#     return float(np.sum(correct_positives) / np.sum(positive_predictions))


# def sharpe_ratio_proxy(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
#     """
#     Calculate Sharpe ratio proxy for predicted returns.
    
#     Args:
#         returns: Array of returns
#         risk_free_rate: Risk-free rate (annualized)
        
#     Returns:
#         Sharpe ratio proxy
#     """
#     excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
#     if np.std(excess_returns) == 0:
#         return 0.0
#     return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))


# def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
#     """
#     Calculate Sortino ratio for predicted returns.
    
#     Args:
#         returns: Array of returns
#         risk_free_rate: Risk-free rate (annualized)
        
#     Returns:
#         Sortino ratio
#     """
#     excess_returns = returns - risk_free_rate / 252
#     downside_returns = excess_returns[excess_returns < 0]
    
#     if len(downside_returns) == 0 or np.std(downside_returns) == 0:
#         return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
#     downside_deviation = np.std(downside_returns)
#     return float(np.mean(excess_returns) / downside_deviation * np.sqrt(252))


# def maximum_drawdown(cumulative_returns: np.ndarray) -> float:
#     """
#     Calculate maximum drawdown from cumulative returns.
    
#     Args:
#         cumulative_returns: Cumulative returns series
        
#     Returns:
#         Maximum drawdown (positive value)
#     """
#     peak = np.maximum.accumulate(cumulative_returns)
#     drawdown = (cumulative_returns - peak) / peak
#     return float(-np.min(drawdown))


# def calmar_ratio(returns: np.ndarray) -> float:
#     """
#     Calculate Calmar ratio (annualized return / maximum drawdown).
    
#     Args:
#         returns: Array of returns
        
#     Returns:
#         Calmar ratio
#     """
#     cumulative_returns = np.cumprod(1 + returns)
#     max_dd = maximum_drawdown(cumulative_returns)
    
#     if max_dd == 0:
#         return float('inf') if np.mean(returns) > 0 else 0.0
    
#     annualized_return = (cumulative_returns[-1] ** (252 / len(returns))) - 1
#     return float(annualized_return / max_dd)


# def information_ratio(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
#     """
#     Calculate information ratio.
    
#     Args:
#         portfolio_returns: Portfolio returns
#         benchmark_returns: Benchmark returns
        
#     Returns:
#         Information ratio
#     """
#     active_returns = portfolio_returns - benchmark_returns
#     tracking_error = np.std(active_returns)
    
#     if tracking_error == 0:
#         return 0.0
    
#     return float(np.mean(active_returns) / tracking_error)


# def value_at_risk(returns: np.ndarray, confidence_level: float = 0.05) -> float:
#     """
#     Calculate Value at Risk (VaR).
    
#     Args:
#         returns: Array of returns
#         confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
#     Returns:
#         VaR value (positive for losses)
#     """
#     return float(-np.percentile(returns, confidence_level * 100))


# def expected_shortfall(returns: np.ndarray, confidence_level: float = 0.05) -> float:
#     """
#     Calculate Expected Shortfall (Conditional VaR).
    
#     Args:
#         returns: Array of returns
#         confidence_level: Confidence level
        
#     Returns:
#         Expected shortfall (positive for losses)
#     """
#     var = value_at_risk(returns, confidence_level)
#     tail_losses = returns[returns <= -var]
    
#     if len(tail_losses) == 0:
#         return var
    
#     return float(-np.mean(tail_losses))


"""
Model evaluator implementations for standardized performance assessment.

This module contains evaluator classes that implement the evaluation protocols
and provide standardized metrics for different modeling tasks.
"""

import numpy as np
from typing import Dict, List

from stock_market_analytics.modeling.model_factory.protocols import ModelEvaluator, QuantileEvaluator
from stock_market_analytics.modeling.model_factory.evaluation.evaluation_functions import (
    pinball_loss_vectorized,
    quantile_coverage,
    quantile_loss_differential,
    interval_score,
    prediction_interval_coverage_probability,
    mean_interval_width,
    normalized_interval_width,
    crps_from_quantiles,
    monotonicity_violation_rate,
    ensure_sorted_unique_quantiles,
    validate_xyq_shapes,
    align_predictions_to_quantiles,
)




# -------------------------
# Evaluator
# -------------------------

class QuantileRegressionEvaluator:
    """
    Evaluator for quantile regression tasks.

    Provides metrics specific to quantile predictions including
    pinball loss, coverage, interval scores, CRPS, and crossing diagnostics.
    """

    def __init__(self, quantiles: List[float]):
        """
        Initialize the quantile regression evaluator.

        Args:
            quantiles: List of quantile levels being predicted
        """
        self.quantiles = ensure_sorted_unique_quantiles(np.array(quantiles, dtype=float))

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        # Intentionally empty for point predictions in a QR context.
        return {}

    def evaluate_quantiles(
        self, 
        y_true: np.ndarray, 
        y_pred_quantiles: np.ndarray, 
        quantiles: List[float]
    ) -> Dict[str, float]:
        """
        Evaluate quantile predictions.
        """
        validate_xyq_shapes(y_true, np.asarray(y_pred_quantiles), np.asarray(quantiles))
        # Align to the provided quantiles (and check they are sane)
        yq, q_sorted, _ = align_predictions_to_quantiles(y_pred_quantiles, np.array(quantiles, dtype=float))

        # Basic metrics
        pinball = pinball_loss_vectorized(y_true, yq, q_sorted)
        cov = quantile_coverage(y_true, yq, q_sorted)
        qld = quantile_loss_differential(y_true, yq, q_sorted)
        crps = crps_from_quantiles(y_true, yq, q_sorted)

        # Crossing diagnostics
        cross_rate, cross_count = monotonicity_violation_rate(yq)

        metrics: Dict[str, float] = {
            'mean_pinball_loss' : float(np.mean(pinball)),
            'median_pinball_loss': float(np.median(pinball)),
            'max_pinball_loss'  : float(np.max(pinball)),
            'mean_coverage'     : float(np.mean(cov)),
            'coverage_deviation': float(np.mean(np.abs(cov - q_sorted))),
            'crps'              : crps,
            'monotonicity_violation_rate': float(cross_rate),
            'monotonicity_violated_rows' : float(cross_count),  # count
        }

        # Per-quantile
        for i, q in enumerate(q_sorted):
            pct = int(round(q * 100))
            metrics[f'pinball_loss_q{pct}'] = float(pinball[i])
            metrics[f'coverage_q{pct}']     = float(cov[i])
            metrics[f'coverage_error_q{pct}'] = float(abs(cov[i] - q))

        # Differential bundle
        metrics.update(qld)
        return metrics

    def evaluate_intervals(
        self, 
        y_true: np.ndarray, 
        y_lower: np.ndarray, 
        y_upper: np.ndarray,
        alpha: float = 0.1
    ) -> Dict[str, float]:
        """
        Evaluate prediction intervals.
        """
        return {
            'interval_score'           : interval_score(y_true, y_lower, y_upper, alpha),
            'coverage_probability'     : prediction_interval_coverage_probability(y_true, y_lower, y_upper),
            'mean_interval_width'      : mean_interval_width(y_lower, y_upper),
            'normalized_interval_width': normalized_interval_width(y_lower, y_upper, y_true),
        }

    def get_metric_names(self) -> List[str]:
        names = [
            'mean_pinball_loss', 'median_pinball_loss', 'max_pinball_loss',
            'mean_coverage', 'coverage_deviation',
            'mean_coverage_error', 'max_coverage_error', 'coverage_bias',
            'crps', 'monotonicity_violation_rate', 'monotonicity_violated_rows',
        ]
        for q in self.quantiles:
            pct = int(round(float(q) * 100))
            names += [
                f'pinball_loss_q{pct}',
                f'coverage_q{pct}',
                f'coverage_error_q{pct}',
            ]
        return names



# class QuantileRegressionEvaluator:
#     """
#     Evaluator for quantile regression tasks.
    
#     Provides metrics specific to quantile predictions including
#     pinball loss, coverage, and interval scores.
#     """
    
#     def __init__(self, quantiles: List[float]):
#         """
#         Initialize the quantile regression evaluator.
        
#         Args:
#             quantiles: List of quantile levels being predicted
#         """
#         self.quantiles = np.array(sorted(quantiles))
        
#     def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """
#         Standard evaluation interface (not used for quantile evaluation).
        
#         Args:
#             y_true: True values
#             y_pred: Point predictions (not used)
            
#         Returns:
#             Empty dictionary
#         """
#         return {}
    
#     def evaluate_quantiles(
#         self, 
#         y_true: np.ndarray, 
#         y_pred_quantiles: np.ndarray, 
#         quantiles: List[float]
#     ) -> Dict[str, float]:
#         """
#         Evaluate quantile predictions.
        
#         Args:
#             y_true: True target values
#             y_pred_quantiles: Predicted quantiles (n_samples, n_quantiles)
#             quantiles: List of quantiles corresponding to predictions
            
#         Returns:
#             Dictionary of quantile-specific metrics
#         """
#         quantiles_array = np.array(quantiles)
        
#         # Basic quantile metrics
#         pinball_losses = pinball_loss_vectorized(y_true, y_pred_quantiles, quantiles_array)
#         coverage = quantile_coverage(y_true, y_pred_quantiles, quantiles_array)
        
#         metrics = {
#             'mean_pinball_loss': float(np.mean(pinball_losses)),
#             'median_pinball_loss': float(np.median(pinball_losses)),
#             'max_pinball_loss': float(np.max(pinball_losses)),
#             'mean_coverage': float(np.mean(coverage)),
#             'coverage_deviation': float(np.mean(np.abs(coverage - quantiles_array))),
#         }
        
#         # Individual quantile metrics
#         for i, q in enumerate(quantiles):
#             metrics[f'pinball_loss_q{int(q*100)}'] = float(pinball_losses[i])
#             metrics[f'coverage_q{int(q*100)}'] = float(coverage[i])
#             metrics[f'coverage_error_q{int(q*100)}'] = float(abs(coverage[i] - q))
        
#         # Differential loss metrics
#         diff_metrics = quantile_loss_differential(y_true, y_pred_quantiles, quantiles_array)
#         metrics.update(diff_metrics)
        
#         return metrics
    
#     def evaluate_intervals(
#         self, 
#         y_true: np.ndarray, 
#         y_lower: np.ndarray, 
#         y_upper: np.ndarray,
#         alpha: float = 0.1
#     ) -> Dict[str, float]:
#         """
#         Evaluate prediction intervals.
        
#         Args:
#             y_true: True values
#             y_lower: Lower bounds of prediction intervals
#             y_upper: Upper bounds of prediction intervals
#             alpha: Miscoverage level
            
#         Returns:
#             Dictionary of interval metrics
#         """
#         metrics = {
#             'interval_score': interval_score(y_true, y_lower, y_upper, alpha),
#             'coverage_probability': prediction_interval_coverage_probability(y_true, y_lower, y_upper),
#             'mean_interval_width': mean_interval_width(y_lower, y_upper),
#             'normalized_interval_width': normalized_interval_width(y_lower, y_upper, y_true),
#         }
        
#         return metrics
    
#     def get_metric_names(self) -> List[str]:
#         """Get names of metrics computed by this evaluator."""
#         base_metrics = [
#             'mean_pinball_loss', 'median_pinball_loss', 'max_pinball_loss',
#             'mean_coverage', 'coverage_deviation', 'mean_coverage_error',
#             'max_coverage_error', 'coverage_bias'
#         ]
        
#         # Add individual quantile metrics
#         for q in self.quantiles:
#             base_metrics.extend([
#                 f'pinball_loss_q{int(q*100)}',
#                 f'coverage_q{int(q*100)}',
#                 f'coverage_error_q{int(q*100)}'
#             ])
        
#         return base_metrics


# class RegressionEvaluator:
#     """
#     Standard evaluator for regression tasks.
    
#     Provides comprehensive evaluation metrics for continuous target predictions.
#     """
    
#     def __init__(self, include_percentage_errors: bool = True):
#         """
#         Initialize the regression evaluator.
        
#         Args:
#             include_percentage_errors: Whether to include MAPE and SMAPE
#         """
#         self.include_percentage_errors = include_percentage_errors
        
#     def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """
#         Evaluate regression predictions.
        
#         Args:
#             y_true: True target values
#             y_pred: Predicted values
            
#         Returns:
#             Dictionary of evaluation metrics
#         """
#         metrics = {
#             'mae': mean_absolute_error(y_true, y_pred),
#             'mse': mean_squared_error(y_true, y_pred),
#             'rmse': root_mean_squared_error(y_true, y_pred),
#         }
        
#         if self.include_percentage_errors:
#             metrics.update({
#                 'mape': mean_absolute_percentage_error(y_true, y_pred),
#                 'smape': symmetric_mean_absolute_percentage_error(y_true, y_pred),
#             })
        
#         return metrics
    
#     def get_metric_names(self) -> List[str]:
#         """Get names of metrics computed by this evaluator."""
#         base_metrics = ['mae', 'mse', 'rmse']
#         if self.include_percentage_errors:
#             base_metrics.extend(['mape', 'smape'])
#         return base_metrics


# class FinancialRegressionEvaluator:
#     """
#     Specialized evaluator for financial regression tasks.
    
#     Includes financial-specific metrics like directional accuracy,
#     Sharpe ratio, and risk metrics.
#     """
    
#     def __init__(self, risk_free_rate: float = 0.02):
#         """
#         Initialize the financial regression evaluator.
        
#         Args:
#             risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
#         """
#         self.risk_free_rate = risk_free_rate
        
#     def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """
#         Evaluate financial regression predictions.
        
#         Args:
#             y_true: True return values
#             y_pred: Predicted return values
            
#         Returns:
#             Dictionary of evaluation metrics
#         """
#         # Basic regression metrics
#         metrics = {
#             'mae': mean_absolute_error(y_true, y_pred),
#             'mse': mean_squared_error(y_true, y_pred),
#             'rmse': root_mean_squared_error(y_true, y_pred),
#         }
        
#         # Financial-specific metrics
#         metrics.update({
#             'directional_accuracy': directional_accuracy(y_true, y_pred),
#             'hit_rate': hit_rate(y_true, y_pred),
#             'sharpe_ratio_pred': sharpe_ratio_proxy(y_pred, self.risk_free_rate),
#             'sharpe_ratio_true': sharpe_ratio_proxy(y_true, self.risk_free_rate),
#             'sortino_ratio_pred': sortino_ratio(y_pred, self.risk_free_rate),
#             'sortino_ratio_true': sortino_ratio(y_true, self.risk_free_rate),
#         })
        
#         # Risk metrics
#         try:
#             metrics.update({
#                 'var_5_pred': value_at_risk(y_pred, 0.05),
#                 'var_5_true': value_at_risk(y_true, 0.05),
#                 'es_5_pred': expected_shortfall(y_pred, 0.05),
#                 'es_5_true': expected_shortfall(y_true, 0.05),
#             })
#         except Exception:
#             # Handle edge cases where risk metrics can't be calculated
#             pass
        
#         # Information ratio (using predictions as active returns vs true as benchmark)
#         try:
#             metrics['information_ratio'] = information_ratio(y_pred, y_true)
#         except Exception:
#             pass
        
#         return metrics
    
#     def get_metric_names(self) -> List[str]:
#         """Get names of metrics computed by this evaluator."""
#         return [
#             'mae', 'mse', 'rmse', 'directional_accuracy', 'hit_rate',
#             'sharpe_ratio_pred', 'sharpe_ratio_true', 'sortino_ratio_pred', 
#             'sortino_ratio_true', 'var_5_pred', 'var_5_true', 'es_5_pred', 
#             'es_5_true', 'information_ratio'
#         ]

# class ClassificationEvaluator:
#     """
#     Standard evaluator for classification tasks.
    
#     Provides common classification metrics including accuracy,
#     precision, recall, and F1-score.
#     """
    
#     def __init__(self, average: str = 'weighted'):
#         """
#         Initialize the classification evaluator.
        
#         Args:
#             average: Averaging strategy for multi-class metrics
#         """
#         self.average = average
        
#     def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """
#         Evaluate classification predictions.
        
#         Args:
#             y_true: True class labels
#             y_pred: Predicted class labels
            
#         Returns:
#             Dictionary of evaluation metrics
#         """
#         from sklearn.metrics import (
#             accuracy_score, precision_score, recall_score, f1_score,
#             confusion_matrix, classification_report
#         )
        
#         metrics = {
#             'accuracy': float(accuracy_score(y_true, y_pred)),
#             'precision': float(precision_score(y_true, y_pred, average=self.average, zero_division=0)),
#             'recall': float(recall_score(y_true, y_pred, average=self.average, zero_division=0)),
#             'f1': float(f1_score(y_true, y_pred, average=self.average, zero_division=0)),
#         }
        
#         # Add per-class metrics for binary classification
#         if len(np.unique(y_true)) == 2:
#             metrics.update({
#                 'precision_positive': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
#                 'recall_positive': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
#                 'f1_positive': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
#                 'precision_negative': float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
#                 'recall_negative': float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
#                 'f1_negative': float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
#             })
        
#         return metrics
    
#     def get_metric_names(self) -> List[str]:
#         """Get names of metrics computed by this evaluator."""
#         base_metrics = ['accuracy', 'precision', 'recall', 'f1']
#         base_metrics.extend([
#             'precision_positive', 'recall_positive', 'f1_positive',
#             'precision_negative', 'recall_negative', 'f1_negative'
#         ])
#         return base_metrics


# class CompositeEvaluator:
#     """
#     Composite evaluator that combines multiple evaluation strategies.
    
#     Useful for comprehensive model assessment using different evaluation perspectives.
#     """
    
#     def __init__(self, evaluators: List[ModelEvaluator], prefixes: List[str] | None = None):
#         """
#         Initialize the composite evaluator.
        
#         Args:
#             evaluators: List of evaluators to combine
#             prefixes: Optional prefixes for metric names to avoid conflicts
#         """
#         self.evaluators = evaluators
#         self.prefixes = prefixes or [f"eval_{i}" for i in range(len(evaluators))]
        
#         if len(self.evaluators) != len(self.prefixes):
#             raise ValueError("Number of evaluators must match number of prefixes")
    
#     def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """
#         Evaluate using all component evaluators.
        
#         Args:
#             y_true: True target values
#             y_pred: Predicted values
            
#         Returns:
#             Combined dictionary of all evaluation metrics
#         """
#         combined_metrics = {}
        
#         for evaluator, prefix in zip(self.evaluators, self.prefixes):
#             evaluator_metrics = evaluator.evaluate(y_true, y_pred)
            
#             # Add prefix to avoid metric name conflicts
#             for metric_name, metric_value in evaluator_metrics.items():
#                 combined_metrics[f"{prefix}_{metric_name}"] = metric_value
        
#         return combined_metrics
    
#     def get_metric_names(self) -> List[str]:
#         """Get names of all metrics computed by component evaluators."""
#         all_metrics = []
        
#         for evaluator, prefix in zip(self.evaluators, self.prefixes):
#             evaluator_metrics = evaluator.get_metric_names()
#             all_metrics.extend([f"{prefix}_{name}" for name in evaluator_metrics])
        
#         return all_metrics


# class BacktestEvaluator:
#     """
#     Specialized evaluator for backtesting trading strategies.
    
#     Evaluates strategy performance using portfolio-level metrics
#     including returns, drawdowns, and risk-adjusted measures.
#     """
    
#     def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001):
#         """
#         Initialize the backtest evaluator.
        
#         Args:
#             initial_capital: Starting portfolio value
#             transaction_cost: Transaction cost as fraction of trade value
#         """
#         self.initial_capital = initial_capital
#         self.transaction_cost = transaction_cost
        
#     def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """
#         Evaluate trading strategy performance.
        
#         Args:
#             y_true: True returns
#             y_pred: Predicted returns (used for position sizing)
            
#         Returns:
#             Dictionary of strategy performance metrics
#         """
#         # Simple strategy: position proportional to prediction strength
#         positions = np.sign(y_pred) * np.abs(y_pred)
#         position_changes = np.diff(np.concatenate([[0], positions]))
        
#         # Calculate strategy returns
#         strategy_returns = positions[:-1] * y_true[1:]  # Lag positions by 1 period
        
#         # Subtract transaction costs
#         transaction_costs = np.abs(position_changes[1:]) * self.transaction_cost
#         net_returns = strategy_returns - transaction_costs
        
#         # Calculate cumulative returns
#         cumulative_returns = np.cumprod(1 + net_returns)
        
#         # Performance metrics
#         metrics = {
#             'total_return': float(cumulative_returns[-1] - 1),
#             'annualized_return': float((cumulative_returns[-1] ** (252 / len(net_returns))) - 1),
#             'volatility': float(np.std(net_returns) * np.sqrt(252)),
#             'sharpe_ratio': sharpe_ratio_proxy(net_returns),
#             'sortino_ratio': sortino_ratio(net_returns),
#             'max_drawdown': maximum_drawdown(cumulative_returns),
#             'calmar_ratio': calmar_ratio(net_returns),
#             'var_5': value_at_risk(net_returns, 0.05),
#             'expected_shortfall_5': expected_shortfall(net_returns, 0.05),
#         }
        
#         # Strategy-specific metrics
#         win_rate = float(np.mean(net_returns > 0))
#         avg_win = float(np.mean(net_returns[net_returns > 0])) if np.any(net_returns > 0) else 0.0
#         avg_loss = float(np.mean(net_returns[net_returns < 0])) if np.any(net_returns < 0) else 0.0
        
#         metrics.update({
#             'win_rate': win_rate,
#             'avg_win': avg_win,
#             'avg_loss': avg_loss,
#             'profit_factor': float(abs(avg_win / avg_loss)) if avg_loss != 0 else float('inf'),
#             'num_trades': float(np.sum(np.abs(position_changes) > 1e-6)),
#         })
        
#         return metrics
    
#     def get_metric_names(self) -> List[str]:
#         """Get names of metrics computed by this evaluator."""
#         return [
#             'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
#             'sortino_ratio', 'max_drawdown', 'calmar_ratio', 'var_5',
#             'expected_shortfall_5', 'win_rate', 'avg_win', 'avg_loss',
#             'profit_factor', 'num_trades'
#         ]
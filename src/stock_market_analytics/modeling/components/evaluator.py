"""
Multi-quantile evaluation component.

This module provides a clean interface for evaluating multi-quantile models,
including conformal prediction, baseline comparison, and comprehensive metrics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from catboost import Pool

from ..utils.metrics import (
    eval_multiquantile, 
    predict_quantiles,
    conformal_adjustment,
    apply_conformal,
    coverage,
    mean_width,
    pinball_loss
)
from ..baseline_predictor import BaselinePredictor

logger = logging.getLogger(__name__)


class MultiQuantileEvaluator:
    """
    Comprehensive evaluation for multi-quantile models.
    
    This class handles model evaluation, conformal prediction calibration,
    baseline comparison, and metric computation in a clean, reusable way.
    """
    
    def __init__(
        self, 
        quantiles: List[float],
        target_coverage: float = 0.8,
        coverage_interval: Tuple[float, float] = (0.1, 0.9)
    ):
        """
        Initialize the evaluator.
        
        Args:
            quantiles: List of quantiles used by the model
            target_coverage: Target coverage for conformal prediction
            coverage_interval: Quantile interval for coverage computation
        """
        self.quantiles = quantiles
        self.target_coverage = target_coverage
        self.coverage_interval = coverage_interval
        
        # Find indices for low, mid, high quantiles
        self.low_idx = 0  # Assume first quantile is lowest
        self.high_idx = len(quantiles) - 1  # Assume last quantile is highest
        self.mid_idx = len(quantiles) // 2  # Middle quantile
        
    def evaluate_model(
        self,
        model,
        validation_pool: Pool,
        return_predictions: bool = False,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on validation data.
        
        Args:
            model: Trained model with predict method
            validation_pool: Validation data as CatBoost Pool
            return_predictions: Whether to return predictions in results
            sample_weight: Optional sample weights for evaluation
            
        Returns:
            Dictionary containing evaluation metrics and optionally predictions
        """
        logger.info("Evaluating model on validation data...")
        
        # Get predictions and true values
        predictions = predict_quantiles(model, validation_pool)
        y_true = validation_pool.get_label()
        
        # Compute comprehensive metrics
        loss, metrics = eval_multiquantile(
            y_true=y_true,
            q_pred=predictions, 
            quantiles=self.quantiles,
            interval=self.coverage_interval,
            sample_weight=sample_weight,
            return_per_quantile=True
        )
        
        result = {
            "loss": loss,
            "metrics": metrics,
            "n_samples": len(y_true)
        }
        
        if return_predictions:
            result["predictions"] = predictions
            result["y_true"] = y_true
            
        return result
        
    def calibrate_conformal(
        self,
        model,
        calibration_data: pd.DataFrame,
        features: List[str],
        target: str
    ) -> float:
        """
        Compute conformal prediction adjustment.
        
        Args:
            model: Trained model
            calibration_data: Calibration dataset
            features: Feature columns
            target: Target column
            
        Returns:
            Conformal adjustment quantile
        """
        logger.info("Computing conformal prediction adjustment...")
        
        X_cal = calibration_data[features]
        y_cal = calibration_data[target].values
        
        # Get quantile predictions
        q_cal = predict_quantiles(model, X_cal)
        
        # Extract low and high quantiles for interval
        q_lo_cal = q_cal[:, self.low_idx]
        q_hi_cal = q_cal[:, self.high_idx]
        
        # Compute conformal adjustment
        alpha = 1 - self.target_coverage
        q_conformal = conformal_adjustment(q_lo_cal, q_hi_cal, y_cal, alpha)
        
        logger.info(f"Conformal adjustment: {q_conformal:.4f}")
        return q_conformal
        
    def evaluate_conformal(
        self,
        model,
        test_data: pd.DataFrame,
        features: List[str],
        target: str,
        q_conformal: float
    ) -> Dict[str, Any]:
        """
        Evaluate model with conformal prediction on test data.
        
        Args:
            model: Trained model
            test_data: Test dataset
            features: Feature columns  
            target: Target column
            q_conformal: Conformal adjustment value
            
        Returns:
            Dictionary with conformal evaluation metrics
        """
        logger.info("Evaluating conformal predictions on test data...")
        
        X_test = test_data[features]
        y_test = test_data[target].values
        
        # Get quantile predictions
        q_test = predict_quantiles(model, X_test)
        
        # Extract quantiles
        q_lo_test = q_test[:, self.low_idx]
        q_hi_test = q_test[:, self.high_idx]
        q_mid_test = q_test[:, self.mid_idx]
        
        # Apply conformal adjustment
        lo_conformal, hi_conformal = apply_conformal(q_lo_test, q_hi_test, q_conformal)
        
        # Compute metrics
        cov = coverage(y_test, lo_conformal, hi_conformal)
        width = mean_width(lo_conformal, hi_conformal)
        pin50 = pinball_loss(y_test, q_mid_test, alpha=0.5)
        
        metrics = {
            "coverage": cov,
            "mean_width": width, 
            "pinball_loss": pin50,
            "target_coverage": self.target_coverage,
            "coverage_gap": abs(cov - self.target_coverage)
        }
        
        logger.info(f"Conformal coverage: {cov:.3f} (target: {self.target_coverage:.3f})")
        logger.info(f"Mean interval width: {width:.3f}")
        
        return {
            "metrics": metrics,
            "predictions": {
                "lower": lo_conformal,
                "upper": hi_conformal, 
                "median": q_mid_test
            },
            "y_true": y_test
        }
        
    def evaluate_baselines(
        self,
        train_pool: Pool,
        validation_pool: Pool,
        baseline_strategies: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate baseline predictors for comparison.
        
        Args:
            train_pool: Training data for fitting baselines
            validation_pool: Validation data for evaluation
            baseline_strategies: List of baseline strategies to evaluate
            
        Returns:
            Dictionary with baseline evaluation results
        """
        if baseline_strategies is None:
            baseline_strategies = [
                "historical_quantiles",
                "random_walk", 
                "random_noise",
                "seasonal_naive"
            ]
            
        logger.info(f"Evaluating {len(baseline_strategies)} baseline strategies...")
        
        baseline_results = {}
        y_true = validation_pool.get_label()
        
        for strategy in baseline_strategies:
            logger.info(f"  Evaluating {strategy} baseline...")
            
            # Create and fit baseline predictor
            baseline = BaselinePredictor(strategy=strategy, random_state=42)
            baseline.fit(train_pool)
            
            # Predict on validation set
            baseline_preds = baseline.predict(validation_pool)
            
            # Evaluate using same metrics as trained model
            loss, metrics = eval_multiquantile(
                y_true=y_true, 
                q_pred=baseline_preds, 
                quantiles=self.quantiles, 
                interval=self.coverage_interval
            )
            
            baseline_results[f"baseline_{strategy}"] = {
                "loss": loss,
                "metrics": metrics,
                "model": baseline
            }
            
        return baseline_results
        
    def compare_with_baselines(
        self,
        trained_metrics: Dict[str, Any],
        baseline_results: Dict[str, Dict[str, Any]],
        test_data: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None,
        q_conformal: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compare trained model performance with baselines.
        
        Args:
            trained_metrics: Metrics from trained model
            baseline_results: Results from baseline evaluation
            test_data: Test data for final comparison (optional)
            features: Feature columns (required if test_data provided)
            q_conformal: Conformal adjustment (for test comparison)
            
        Returns:
            Dictionary with comparative analysis
        """
        logger.info("Comparing trained model with baselines...")
        
        comparative_metrics = {}
        
        # If test data provided, evaluate baselines on test set
        if test_data is not None and features is not None:
            logger.info("Evaluating baselines on test set...")
            
            X_test = test_data[features]
            y_test = test_data[self.target].values if hasattr(self, 'target') else test_data.iloc[:, -1].values
            
            baseline_test_results = {}
            
            for strategy_name, baseline_info in baseline_results.items():
                baseline_model = baseline_info["model"]
                
                # Predict quantiles on test set
                q_test_baseline = baseline_model.predict(X_test)
                
                # Apply conformal adjustment if provided
                if q_conformal is not None:
                    q_lo_baseline = q_test_baseline[:, self.low_idx]
                    q_hi_baseline = q_test_baseline[:, self.high_idx]
                    lo_conformal, hi_conformal = apply_conformal(
                        q_lo_baseline, q_hi_baseline, q_conformal
                    )
                    q_mid_baseline = q_test_baseline[:, self.mid_idx]
                    
                    # Compute test metrics
                    cov_baseline = coverage(y_test, lo_conformal, hi_conformal)
                    width_baseline = mean_width(lo_conformal, hi_conformal)
                    pin50_baseline = pinball_loss(y_test, q_mid_baseline, alpha=0.5)
                    
                    baseline_test_results[strategy_name] = {
                        "coverage": cov_baseline,
                        "mean_width": width_baseline,
                        "pinball_loss": pin50_baseline,
                    }
                    
            # Compare with trained model
            for strategy_name, baseline_metrics in baseline_test_results.items():
                for metric_name in ["coverage", "mean_width", "pinball_loss"]:
                    if metric_name in trained_metrics:
                        trained_value = trained_metrics[metric_name]
                        baseline_value = baseline_metrics[metric_name]
                        
                        # Calculate improvement (positive means trained model is better)
                        if metric_name == "coverage":
                            # For coverage, closer to target is better
                            trained_diff = abs(trained_value - self.target_coverage)
                            baseline_diff = abs(baseline_value - self.target_coverage)
                            improvement = baseline_diff - trained_diff
                        elif metric_name in ["mean_width", "pinball_loss"]:
                            # For width and pinball loss, smaller is better
                            improvement = baseline_value - trained_value
                        else:
                            improvement = trained_value - baseline_value
                            
                        comparative_metrics[f"{strategy_name}_{metric_name}_improvement"] = improvement
                        comparative_metrics[f"{strategy_name}_{metric_name}"] = baseline_value
                        
        return {
            "comparative_metrics": comparative_metrics,
            "baseline_test_results": baseline_test_results if test_data is not None else {},
            "summary": self._create_comparison_summary(trained_metrics, comparative_metrics)
        }
        
    def _create_comparison_summary(
        self, 
        trained_metrics: Dict[str, Any], 
        comparative_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a summary of model vs baseline performance."""
        summary = {
            "better_than_baselines": {},
            "overall_performance": "good"
        }
        
        # Check how many baselines the trained model beats
        improvement_keys = [k for k in comparative_metrics.keys() if k.endswith("_improvement")]
        
        for key in improvement_keys:
            improvement = comparative_metrics[key]
            baseline_name = key.replace("_improvement", "").split("_")[1]  # Extract strategy name
            metric_name = "_".join(key.replace("_improvement", "").split("_")[2:])  # Extract metric name
            
            if baseline_name not in summary["better_than_baselines"]:
                summary["better_than_baselines"][baseline_name] = 0
                
            if improvement > 0:
                summary["better_than_baselines"][baseline_name] += 1
                
        return summary
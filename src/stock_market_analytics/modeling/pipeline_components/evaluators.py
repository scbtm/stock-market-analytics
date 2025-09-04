from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from stock_market_analytics.modeling.pipeline_components.functions import (
    eval_multiquantile,
    coverage,
    mean_width,
    pinball_loss
)
from stock_market_analytics.modeling.pipeline_components.configs import modeling_config

class ModelEvaluator:
    """
    Evaluator for stock market analytics models that leverages the pipeline architecture.
    
    This class provides evaluation capabilities for both raw quantile predictions
    and calibrated predictions (conformal bounds). It focuses purely on evaluation
    and does NOT perform any calibration - that is handled by the ConformalCalibrator.
    
    Key methods:
    - evaluate_training(): Evaluate raw quantile predictions during training
    - evaluate_calibrated_predictions(): Evaluate already-calibrated bounds
    - evaluate_quantile_predictions(): Alias for evaluate_training()
    - full_evaluation(): Evaluate both raw and calibrated pipelines
    """
    
    def __init__(
        self, 
        quantiles: list[float] = None,
        interval: tuple[float, float] = None,
        target_coverage: float = None
    ):
        self.quantiles = quantiles or modeling_config["QUANTILES"]
        self.interval = interval or (modeling_config["LOW"], modeling_config["HIGH"])  
        self.target_coverage = target_coverage or modeling_config["TARGET_COVERAGE"]
        
        # Convert interval indices to alpha values if needed
        if isinstance(self.interval[0], int):
            self.interval = (self.quantiles[self.interval[0]], self.quantiles[self.interval[1]])
    
    def evaluate_training(
        self,
        pipeline: Pipeline,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.DataFrame | np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        lambda_cross: float = 0.0,
        return_per_quantile: bool = False
    ) -> tuple[float, dict[str, Any]]:
        """
        Evaluate model performance during training using validation data.
        
        Args:
            pipeline: Fitted sklearn pipeline containing the model
            X_val: Validation features
            y_val: Validation targets
            sample_weight: Optional sample weights for evaluation
            lambda_cross: Penalty for quantile crossing
            return_per_quantile: Whether to return per-quantile metrics
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Get predictions from pipeline
        q_pred = pipeline.predict(X_val)
        
        if hasattr(y_val, 'values'):
            y_true = y_val.values
        else:
            y_true = np.asarray(y_val)
        
        return eval_multiquantile(
            y_true=y_true,
            q_pred=q_pred,
            quantiles=self.quantiles,
            interval=self.interval,
            sample_weight=sample_weight,
            lambda_cross=lambda_cross,
            return_per_quantile=return_per_quantile
        )
    
    def evaluate_calibrated_predictions(
        self,
        calibrated_predictions: np.ndarray,
        y_true: pd.DataFrame | np.ndarray,
        mid_predictions: Optional[np.ndarray] = None
    ) -> dict[str, Any]:
        """
        Evaluate calibrated predictions (conformal bounds).
        
        This method evaluates predictions that are already calibrated,
        typically from a calibrated pipeline that returns bounds.
        
        Args:
            calibrated_predictions: Calibrated bounds, shape (n_samples, 2)
                                  Column 0: lower bounds, Column 1: upper bounds
            y_true: True targets
            mid_predictions: Optional median predictions for pinball loss
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Convert inputs to arrays
        bounds = np.asarray(calibrated_predictions)
        if hasattr(y_true, 'values'):
            y_array = y_true.values.ravel()
        else:
            y_array = np.asarray(y_true).ravel()
        
        # Validate inputs
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"calibrated_predictions must have shape (n_samples, 2), got {bounds.shape}")
        
        if len(y_array) != bounds.shape[0]:
            raise ValueError(f"Mismatched samples: predictions {bounds.shape[0]}, targets {len(y_array)}")
        
        # Extract bounds
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        
        # Calculate coverage and width metrics
        cov = coverage(y_array, lower_bounds, upper_bounds)
        width = mean_width(lower_bounds, upper_bounds)
        
        results = {
            "coverage": cov,
            "mean_width": width
        }
        
        # Add pinball loss if median predictions provided
        if mid_predictions is not None:
            mid_array = np.asarray(mid_predictions).ravel()
            if len(mid_array) == len(y_array):
                pin50 = pinball_loss(y_array, mid_array, alpha=0.5)
                results["pinball_loss"] = pin50
        
        return results
    
    def evaluate_quantile_predictions(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        lambda_cross: float = 0.0,
        return_per_quantile: bool = False
    ) -> tuple[float, dict[str, Any]]:
        """
        Evaluate raw quantile predictions from a pipeline.
        
        This method evaluates the raw quantile predictions before any
        conformal calibration is applied.
        
        Args:
            pipeline: Pipeline that produces quantile predictions
            X: Features
            y: Targets
            sample_weight: Optional sample weights
            lambda_cross: Penalty for quantile crossing
            return_per_quantile: Whether to return per-quantile metrics
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        return self.evaluate_training(
            pipeline, X, y, sample_weight, lambda_cross, return_per_quantile
        )
    
    def full_evaluation(
        self,
        raw_pipeline: Pipeline,
        calibrated_pipeline: Pipeline,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.DataFrame | np.ndarray,
        return_predictions: bool = False
    ) -> dict[str, Any]:
        """
        Perform complete evaluation of both raw and calibrated pipelines.
        
        Args:
            raw_pipeline: Pipeline producing raw quantile predictions
            calibrated_pipeline: Pipeline producing calibrated bounds
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            return_predictions: Whether to include predictions in results
            
        Returns:
            Complete evaluation results dictionary
        """
        # Evaluate raw quantile predictions
        training_loss, training_metrics = self.evaluate_training(raw_pipeline, X_val, y_val)
        
        # Evaluate calibrated predictions
        calibrated_bounds = calibrated_pipeline.predict(X_test)
        
        # Get median predictions for pinball loss (from raw pipeline)
        raw_predictions = raw_pipeline.predict(X_test)
        mid_idx = modeling_config["MID"]
        median_predictions = raw_predictions[:, mid_idx] if raw_predictions.ndim > 1 else None
        
        calibrated_results = self.evaluate_calibrated_predictions(
            calibrated_bounds, y_test, median_predictions
        )
        
        results = {
            "training": {
                "loss": training_loss,
                "metrics": training_metrics
            },
            "calibrated": calibrated_results
        }
        
        if return_predictions:
            results["predictions"] = {
                "raw_quantiles": raw_predictions,
                "calibrated_bounds": calibrated_bounds
            }
        
        return results


class EvaluationReport:
    """
    Helper class to generate formatted evaluation reports.
    """
    
    @staticmethod
    def format_metrics(metrics: dict[str, Any], title: str = "Evaluation Metrics") -> str:
        """Format metrics dictionary into a readable string."""
        lines = [f"=� {title}", "=" * (len(title) + 4)]
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"\n{key.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        lines.append(f"  {sub_key}: {sub_value:.4f}")
                    else:
                        lines.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, float):
                lines.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def print_summary(evaluation_results: dict[str, Any]) -> None:
        """Print a formatted summary of evaluation results."""
        if "training" in evaluation_results:
            print(EvaluationReport.format_metrics(
                evaluation_results["training"]["metrics"], 
                "Training Metrics"
            ))
            
        if "conformal" in evaluation_results:
            print("\n" + EvaluationReport.format_metrics(
                evaluation_results["conformal"], 
                "Conformal Evaluation"
            ))
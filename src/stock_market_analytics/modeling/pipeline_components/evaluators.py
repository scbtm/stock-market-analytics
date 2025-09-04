from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from stock_market_analytics.modeling.pipeline_components.functions import (
    eval_multiquantile,
    conformal_adjustment,
    apply_conformal,
    coverage,
    mean_width,
    pinball_loss
)
from stock_market_analytics.modeling.pipeline_components.configs import modeling_config

class ModelEvaluator:
    """
    Evaluator for stock market analytics models that leverages the pipeline architecture.
    
    This class provides a clean interface for both training-time and conformal evaluation
    while maintaining flexibility through the pipeline_components architecture.
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
    
    def evaluate_conformal(
        self,
        pipeline: Pipeline,
        X_cal: pd.DataFrame | np.ndarray,
        y_cal: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.DataFrame | np.ndarray,
        low_idx: int = None,
        high_idx: int = None,
        mid_idx: int = None
    ) -> dict[str, Any]:
        """
        Perform conformal prediction evaluation.
        
        Args:
            pipeline: Fitted sklearn pipeline containing the model
            X_cal: Calibration features
            y_cal: Calibration targets
            X_test: Test features
            y_test: Test targets
            low_idx: Index of low quantile (defaults to modeling_config["LOW"])
            high_idx: Index of high quantile (defaults to modeling_config["HIGH"])
            mid_idx: Index of median quantile (defaults to modeling_config["MID"])
            
        Returns:
            Dictionary containing conformal evaluation metrics
        """
        # Default indices from config
        low_idx = low_idx if low_idx is not None else modeling_config["LOW"]
        high_idx = high_idx if high_idx is not None else modeling_config["HIGH"]
        mid_idx = mid_idx if mid_idx is not None else modeling_config["MID"]
        
        # Convert targets to arrays
        if hasattr(y_cal, 'values'):
            y_cal_array = y_cal.values
        else:
            y_cal_array = np.asarray(y_cal)
            
        if hasattr(y_test, 'values'):
            y_test_array = y_test.values
        else:
            y_test_array = np.asarray(y_test)
        
        # Get predictions
        q_cal = pipeline.predict(X_cal)
        q_test = pipeline.predict(X_test)
        
        # Extract specific quantiles for conformal adjustment
        qlo_cal, qhi_cal = q_cal[:, low_idx], q_cal[:, high_idx]
        qlo_test, qhi_test = q_test[:, low_idx], q_test[:, high_idx]
        
        # Conformal adjustment
        alpha = 1 - self.target_coverage
        q_conformal = conformal_adjustment(qlo_cal, qhi_cal, y_cal_array, alpha=alpha)
        
        # Apply conformal adjustment to test set
        lo_cqr, hi_cqr = apply_conformal(qlo_test, qhi_test, q_conformal)
        med_pred = q_test[:, mid_idx]
        
        # Calculate metrics
        cov = coverage(y_test_array, lo_cqr, hi_cqr)
        width = mean_width(lo_cqr, hi_cqr)
        pin50 = pinball_loss(y_test_array, med_pred, alpha=0.5)
        
        return {
            "coverage": cov,
            "mean_width": width,
            "pinball_loss": pin50,
            "conformal_quantile": q_conformal,
            "predictions": {
                "lower_bound": lo_cqr,
                "upper_bound": hi_cqr,
                "median": med_pred
            }
        }
    
    def full_evaluation(
        self,
        pipeline: Pipeline,
        modeling_datasets: dict[str, Any],
        return_predictions: bool = False
    ) -> dict[str, Any]:
        """
        Perform complete evaluation including both training and conformal steps.
        
        Args:
            pipeline: Fitted sklearn pipeline
            modeling_datasets: Dictionary containing train/val/test splits
            return_predictions: Whether to include predictions in results
            
        Returns:
            Complete evaluation results dictionary
        """
        # Training evaluation
        X_val, y_val = modeling_datasets["xval"], modeling_datasets["yval"]
        training_loss, training_metrics = self.evaluate_training(pipeline, X_val, y_val)
        
        # Conformal evaluation
        X_cal, y_cal = modeling_datasets["xval"], modeling_datasets["yval"] 
        X_test, y_test = modeling_datasets["xtest"], modeling_datasets["ytest"]
        
        conformal_results = self.evaluate_conformal(
            pipeline, X_cal, y_cal, X_test, y_test
        )
        
        results = {
            "training": {
                "loss": training_loss,
                "metrics": training_metrics
            },
            "conformal": {
                key: value for key, value in conformal_results.items() 
                if key != "predictions"
            }
        }
        
        if return_predictions:
            results["predictions"] = conformal_results["predictions"]
        
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
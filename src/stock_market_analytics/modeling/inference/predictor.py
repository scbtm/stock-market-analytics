"""
Production predictor for multi-quantile stock return predictions.

This module provides a production-ready predictor that combines the trained
model with conformal prediction adjustments for uncertainty quantification.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib

from ..components import CatBoostMultiQuantileModel, DataProcessor, MultiQuantileEvaluator
from ..config import ConfigManager

logger = logging.getLogger(__name__)


class ProductionPredictor:
    """
    Production-ready predictor for multi-quantile stock return forecasting.
    
    This class encapsulates a trained model, conformal adjustment, and all
    preprocessing logic needed to make predictions in production environments.
    """
    
    def __init__(
        self,
        model: CatBoostMultiQuantileModel,
        config: ConfigManager,
        conformal_adjustment: Optional[float] = None,
        model_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize production predictor.
        
        Args:
            model: Trained CatBoost multi-quantile model
            config: Configuration manager with all settings
            conformal_adjustment: Conformal prediction adjustment value
            model_metadata: Optional metadata about the model training
        """
        self.model = model
        self.config = config
        self.conformal_adjustment = conformal_adjustment
        self.model_metadata = model_metadata or {}
        
        # Initialize components
        self.data_processor = DataProcessor(
            features=config.data.features,
            target=config.data.target
        )
        
        self.evaluator = MultiQuantileEvaluator(
            quantiles=config.model.catboost.quantiles,
            target_coverage=config.evaluation.target_coverage,
            coverage_interval=config.evaluation.coverage_interval
        )
        
    def predict(
        self,
        data: Union[pd.DataFrame, str, Path],
        return_intervals: bool = True,
        apply_conformal: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Make multi-quantile predictions on new data.
        
        Args:
            data: Input data (DataFrame or path to data file)
            return_intervals: Whether to return prediction intervals
            apply_conformal: Whether to apply conformal adjustment
            
        Returns:
            Dictionary with predictions and optionally intervals
        """
        # Load data if path provided
        if isinstance(data, (str, Path)):
            data = self._load_data(data)
            
        # Validate and prepare data
        self._validate_input_data(data)
        data_clean = self.data_processor.clean_data(data)
        
        logger.info(f"Making predictions on {len(data_clean)} samples")
        
        # Extract features
        X = data_clean[self.config.data.features]
        
        # Get quantile predictions
        quantile_predictions = self.model.predict(X)
        
        result = {
            "quantile_predictions": quantile_predictions,
            "quantiles": self.config.model.catboost.quantiles,
            "feature_names": self.config.data.features
        }
        
        # Add prediction intervals if requested
        if return_intervals:
            intervals = self._compute_prediction_intervals(
                quantile_predictions, 
                apply_conformal=apply_conformal
            )
            result.update(intervals)
            
        # Add metadata
        if hasattr(data_clean, 'index'):
            result["index"] = data_clean.index
        if "symbol" in data_clean.columns:
            result["symbols"] = data_clean["symbol"].values
        if "date" in data_clean.columns:
            result["dates"] = data_clean["date"].values
            
        return result
        
    def predict_single_sample(
        self,
        features: Dict[str, Any],
        apply_conformal: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction for a single sample.
        
        Args:
            features: Dictionary with feature values
            apply_conformal: Whether to apply conformal adjustment
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Make prediction
        result = self.predict(df, apply_conformal=apply_conformal)
        
        # Return single sample results
        single_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray) and len(value) == 1:
                single_result[key] = value[0]
            elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                single_result[key] = value
            else:
                single_result[key] = value
                
        return single_result
        
    def _load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from file path."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        if data_path.suffix == ".parquet":
            return pd.read_parquet(data_path)
        elif data_path.suffix == ".csv":
            return pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
            
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data contains required features."""
        missing_features = set(self.config.data.features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Check for any null values in features
        feature_nulls = data[self.config.data.features].isnull().sum()
        features_with_nulls = feature_nulls[feature_nulls > 0]
        
        if len(features_with_nulls) > 0:
            logger.warning(f"Features with null values: {features_with_nulls.to_dict()}")
            
    def _compute_prediction_intervals(
        self,
        quantile_predictions: np.ndarray,
        apply_conformal: bool = True
    ) -> Dict[str, np.ndarray]:
        """Compute prediction intervals from quantile predictions."""
        result = {}
        
        # Extract key quantiles
        low_quantile = quantile_predictions[:, self.evaluator.low_idx]
        high_quantile = quantile_predictions[:, self.evaluator.high_idx]  
        median_quantile = quantile_predictions[:, self.evaluator.mid_idx]
        
        result["median_prediction"] = median_quantile
        result["prediction_intervals"] = {
            "lower": low_quantile,
            "upper": high_quantile
        }
        
        # Apply conformal adjustment if available and requested
        if apply_conformal and self.conformal_adjustment is not None:
            from ..utils.metrics import apply_conformal
            
            lower_adjusted, upper_adjusted = apply_conformal(
                low_quantile, high_quantile, self.conformal_adjustment
            )
            
            result["conformal_intervals"] = {
                "lower": lower_adjusted,
                "upper": upper_adjusted,
                "adjustment": self.conformal_adjustment
            }
            
        return result
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        info = {
            "model_type": "CatBoost Multi-Quantile",
            "quantiles": self.config.model.catboost.quantiles,
            "features": self.config.data.features,
            "n_features": len(self.config.data.features),
            "target": self.config.data.target,
            "conformal_adjustment": self.conformal_adjustment,
            "best_iteration": self.model.best_iteration
        }
        
        # Add feature importances if available
        if self.model.feature_importances_ is not None:
            feature_importance_dict = dict(zip(
                self.config.data.features,
                self.model.feature_importances_
            ))
            # Sort by importance
            info["feature_importances"] = dict(
                sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
        # Add training metadata if available
        if self.model_metadata:
            info["training_metadata"] = self.model_metadata
            
        return info
        
    def save(self, save_path: Union[str, Path]) -> None:
        """
        Save the complete predictor to disk.
        
        Args:
            save_path: Path where to save the predictor
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using joblib for complete serialization
        predictor_data = {
            "model": self.model,
            "config": self.config,
            "conformal_adjustment": self.conformal_adjustment,
            "model_metadata": self.model_metadata
        }
        
        joblib.dump(predictor_data, save_path)
        logger.info(f"Predictor saved to {save_path}")
        
    @classmethod
    def load(cls, save_path: Union[str, Path]) -> "ProductionPredictor":
        """
        Load a saved predictor from disk.
        
        Args:
            save_path: Path to the saved predictor
            
        Returns:
            Loaded ProductionPredictor instance
        """
        save_path = Path(save_path)
        
        if not save_path.exists():
            raise FileNotFoundError(f"Predictor file not found: {save_path}")
            
        predictor_data = joblib.load(save_path)
        
        return cls(
            model=predictor_data["model"],
            config=predictor_data["config"],
            conformal_adjustment=predictor_data.get("conformal_adjustment"),
            model_metadata=predictor_data.get("model_metadata", {})
        )
        
    def __repr__(self) -> str:
        """String representation of the predictor."""
        return (f"ProductionPredictor(quantiles={self.config.model.catboost.quantiles}, "
                f"features={len(self.config.data.features)}, "
                f"conformal={self.conformal_adjustment is not None})")
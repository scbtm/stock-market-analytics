"""
Concrete implementations of protocol-compliant factories and splitters.

This module provides implementations for:
- Data splitting strategies (time-based, k-fold, etc.)
- Model factories for creating different model types
- Task configurations for different modeling tasks
"""

from typing import Any, Mapping
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator

from stock_market_analytics.config import config
from .protocols import (
    DataSplitterProtocol,
    DataSplit,
    ModelFactoryProtocol,
    TaskConfigProtocol,
    TaskType,
)
from .predictors import CatBoostMultiQuantileModel
from .naive_baselines import HistoricalQuantileBaseline
from .evaluators import QuantileRegressionEvaluator
from .calibrators import QuantileIntervalCalibrator

NDArrayF = npt.NDArray[np.float64]


class TimeSeriesDataSplitter(DataSplitterProtocol):
    """
    Time-based data splitter for financial time series.
    
    Creates chronological train/validation/test splits to prevent data leakage
    in time series modeling.
    """
    
    def __init__(
        self,
        time_span: int | None = None,
        split_names: list[str] | None = None,
    ):
        """
        Initialize time series splitter.
        
        Args:
            time_span: Number of days for validation and test sets
            split_names: Custom split names (default: ["train", "validation", "test"])
        """
        self.time_span = time_span or config.modeling.time_span
        self._split_names = split_names or ["train", "validation", "test"]
    
    @property
    def split_names(self) -> list[str]:
        """Return the expected split names this splitter produces."""
        return self._split_names
    
    def split_data(
        self,
        data: Any,
        target: Any | None = None,
        groups: Any | None = None,
        **kwargs: Any
    ) -> list[DataSplit]:
        """
        Split data chronologically into train/validation/test sets.
        
        Args:
            data: DataFrame with 'date' column for temporal ordering
            target: Unused for time-based splits
            groups: Unused for time-based splits
            **kwargs: Additional parameters (time_span override)
            
        Returns:
            List of DataSplit objects with indices for each split
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("TimeSeriesDataSplitter requires pandas DataFrame with 'date' column")
        
        if 'date' not in data.columns:
            raise ValueError("Data must contain 'date' column for time-based splitting")
        
        time_span = kwargs.get('time_span', self.time_span)
        
        # Sort by date to ensure chronological order
        data_sorted = data.sort_values('date').reset_index(drop=True)
        
        # Calculate split boundaries
        max_date = data_sorted['date'].max()
        test_start_date = max_date - pd.Timedelta(days=time_span)
        val_start_date = test_start_date - pd.Timedelta(days=time_span)
        
        # Create splits
        train_mask = data_sorted['date'] < val_start_date
        val_mask = (data_sorted['date'] >= val_start_date) & (data_sorted['date'] < test_start_date)
        test_mask = data_sorted['date'] >= test_start_date
        
        # Create metadata
        train_metadata = {
            "start_date": data_sorted.loc[train_mask, 'date'].min(),
            "end_date": data_sorted.loc[train_mask, 'date'].max(),
            "n_samples": train_mask.sum(),
        }
        
        val_metadata = {
            "start_date": data_sorted.loc[val_mask, 'date'].min(),
            "end_date": data_sorted.loc[val_mask, 'date'].max(),
            "n_samples": val_mask.sum(),
        }
        
        test_metadata = {
            "start_date": data_sorted.loc[test_mask, 'date'].min(),
            "end_date": data_sorted.loc[test_mask, 'date'].max(),
            "n_samples": test_mask.sum(),
        }
        
        return [
            DataSplit("train", np.where(train_mask)[0], train_metadata),
            DataSplit("validation", np.where(val_mask)[0], val_metadata),
            DataSplit("test", np.where(test_mask)[0], test_metadata),
        ]


class QuantileRegressionModelFactory(ModelFactoryProtocol):
    """
    Factory for creating quantile regression models.
    
    Supports various model types with consistent interfaces for experimentation.
    """
    
    def __init__(self, quantiles: list[float] | None = None):
        """
        Initialize model factory.
        
        Args:
            quantiles: Default quantiles for all models
        """
        self.quantiles = quantiles or config.modeling.quantiles
    
    def get_available_models(self) -> list[str]:
        """Return list of available model types."""
        return ["catboost", "historical"]
    
    def create_model(
        self, 
        model_type: str, 
        **kwargs: Any
    ) -> BaseEstimator:
        """
        Create a model of the specified type.
        
        Args:
            model_type: Model identifier ("catboost", "historical", etc.)
            **kwargs: Model-specific parameters
            
        Returns:
            Unfitted BaseEstimator implementing SupportsPredictQuantiles
        """
        # Merge default quantiles with kwargs
        model_params = {"quantiles": self.quantiles}
        model_params.update(kwargs)
        
        if model_type == "catboost":
            # Add CatBoost-specific parameters from config
            cb_params = config.modeling.cb_model_params.copy()
            cb_params.update(model_params)
            return CatBoostMultiQuantileModel(**cb_params)
        
        elif model_type == "historical":
            return HistoricalQuantileBaseline(**model_params)
        
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Available options: {self.get_available_models()}"
            )
    
    def get_model_info(self, model_type: str) -> Mapping[str, Any]:
        """Get metadata about a specific model type."""
        if model_type == "catboost":
            return {
                "name": "CatBoost Multi-Quantile Regressor",
                "type": "gradient_boosting",
                "supports_early_stopping": True,
                "supports_categorical": True,
                "default_params": config.modeling.cb_model_params,
            }
        
        elif model_type == "historical":
            return {
                "name": "Historical Quantile Baseline",
                "type": "naive_baseline",
                "supports_early_stopping": False,
                "supports_categorical": False,
                "default_params": {},
            }
        
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Available options: {self.get_available_models()}"
            )


class QuantileRegressionTaskConfig(TaskConfigProtocol):
    """
    Task configuration for quantile regression modeling.
    
    Provides defaults and factory methods for quantile regression tasks.
    """
    
    @property
    def task_type(self) -> TaskType:
        """Return the task type."""
        return "quantile_regression"
    
    @property
    def primary_metric_name(self) -> str:
        """Return the primary metric for optimization."""
        return "pinball_mean"
    
    @property
    def default_quantiles(self) -> list[float]:
        """Return default quantiles for this task."""
        return config.modeling.quantiles
    
    @property
    def prefers_calibration(self) -> bool:
        """Whether this task benefits from calibration."""
        return True
    
    def make_default_evaluator(self) -> QuantileRegressionEvaluator:
        """Create a default evaluator for this task."""
        return QuantileRegressionEvaluator(
            quantiles=self.default_quantiles,
            target_coverage=config.modeling.target_coverage,
        )
    
    def make_default_calibrator(self) -> QuantileIntervalCalibrator:
        """Create a default calibrator for this task."""
        return QuantileIntervalCalibrator(
            target_coverage=config.modeling.target_coverage,
            low_quantile=self.default_quantiles[0],    # First quantile
            high_quantile=self.default_quantiles[-1],  # Last quantile
        )
"""
Data splitting strategies for stock market analytics modeling.

This module provides various data splitting implementations including
time-based splits for financial time series.
"""

from typing import Any
import numpy as np
import numpy.typing as npt
import pandas as pd

from stock_market_analytics.config import config
from stock_market_analytics.modeling.model_factory.protocols import (
    DataSplitterProtocol,
    DataSplit,
)

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
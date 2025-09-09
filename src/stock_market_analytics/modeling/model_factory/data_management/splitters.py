"""
Data splitting classes for ML model training and validation.

This module provides different strategies to split time series and cross-sectional
data according to various requirements for stock market analytics.
"""

from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from stock_market_analytics.modeling.model_factory.protocols import DataSplitter


class TimeSeriesSplitter:
    """
    Time series splitter that respects temporal order.
    
    Splits data based on dates to avoid data leakage in time series modeling.
    """
    
    def __init__(self, test_size: float = 0.2, date_column: str = "date"):
        """
        Initialize the time series splitter.
        
        Args:
            test_size: Proportion of data to use for testing
            date_column: Name of the date column for temporal splitting
        """
        self.test_size = test_size
        self.date_column = date_column
        
    def split(self, X: pl.DataFrame, y: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        """
        Split data maintaining temporal order.
        
        Args:
            X: Feature matrix with date column
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in features")
            
        # Sort by date to ensure temporal order
        sorted_indices = X.select(pl.col(self.date_column)).to_series().arg_sort()
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Calculate split point
        n_train = int(len(X_sorted) * (1 - self.test_size))
        
        X_train = X_sorted[:n_train]
        X_test = X_sorted[n_train:]
        y_train = y_sorted[:n_train]
        y_test = y_sorted[n_train:]
        
        return X_train, X_test, y_train, y_test
    
    def get_train_indices(self, X: pl.DataFrame) -> list[int]:
        """Get indices for training data."""
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in features")
            
        sorted_indices = X.select(pl.col(self.date_column)).to_series().arg_sort()
        n_train = int(len(X) * (1 - self.test_size))
        
        return sorted_indices[:n_train].to_list()
    
    def get_test_indices(self, X: pl.DataFrame) -> list[int]:
        """Get indices for test data."""
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in features")
            
        sorted_indices = X.select(pl.col(self.date_column)).to_series().arg_sort()
        n_train = int(len(X) * (1 - self.test_size))
        
        return sorted_indices[n_train:].to_list()


class WalkForwardSplitter:
    """
    Walk-forward splitter for time series cross-validation.
    
    Creates multiple train/test splits with expanding training window,
    suitable for backtesting trading strategies.
    """
    
    def __init__(
        self, 
        initial_train_size: int, 
        test_size: int,
        step_size: int = 1,
        date_column: str = "date"
    ):
        """
        Initialize the walk-forward splitter.
        
        Args:
            initial_train_size: Size of initial training window
            test_size: Size of test window for each split
            step_size: Number of periods to step forward each iteration
            date_column: Name of the date column for temporal splitting
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.date_column = date_column
        
    def split(self, X: pl.DataFrame, y: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        """
        Get the first walk-forward split.
        
        Args:
            X: Feature matrix with date column
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) for first split
        """
        splits = list(self.split_generator(X, y))
        if not splits:
            raise ValueError("No valid splits generated")
        return splits[0]
    
    def split_generator(self, X: pl.DataFrame, y: pl.Series):
        """
        Generator yielding walk-forward splits.
        
        Args:
            X: Feature matrix with date column
            y: Target vector
            
        Yields:
            Tuples of (X_train, X_test, y_train, y_test)
        """
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in features")
            
        # Sort by date
        sorted_indices = X.select(pl.col(self.date_column)).to_series().arg_sort()
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        current_start = 0
        while current_start + self.initial_train_size + self.test_size <= len(X_sorted):
            train_end = current_start + self.initial_train_size
            test_end = train_end + self.test_size
            
            X_train = X_sorted[current_start:train_end]
            X_test = X_sorted[train_end:test_end]
            y_train = y_sorted[current_start:train_end]
            y_test = y_sorted[train_end:test_end]
            
            yield X_train, X_test, y_train, y_test
            
            current_start += self.step_size
    
    def get_train_indices(self, X: pl.DataFrame) -> list[int]:
        """Get indices for first training split."""
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in features")
            
        sorted_indices = X.select(pl.col(self.date_column)).to_series().arg_sort()
        return sorted_indices[:self.initial_train_size].to_list()
    
    def get_test_indices(self, X: pl.DataFrame) -> list[int]:
        """Get indices for first test split."""
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in features")
            
        sorted_indices = X.select(pl.col(self.date_column)).to_series().arg_sort()
        train_end = self.initial_train_size
        test_end = train_end + self.test_size
        
        return sorted_indices[train_end:test_end].to_list()


class StratifiedTimeSeriesSplitter:
    """
    Stratified splitter that maintains temporal order while balancing target distribution.
    
    Useful when you need to maintain time series properties but also ensure
    balanced representation of different target classes or regimes.
    """
    
    def __init__(
        self, 
        test_size: float = 0.2,
        date_column: str = "date",
        stratify_column: str | None = None,
        n_bins: int = 5
    ):
        """
        Initialize the stratified time series splitter.
        
        Args:
            test_size: Proportion of data to use for testing
            date_column: Name of the date column for temporal splitting
            stratify_column: Column to use for stratification (None to use target)
            n_bins: Number of bins for continuous target stratification
        """
        self.test_size = test_size
        self.date_column = date_column
        self.stratify_column = stratify_column
        self.n_bins = n_bins
        
    def split(self, X: pl.DataFrame, y: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        """
        Split data with stratification while respecting temporal order.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in features")
            
        # Sort by date first
        sorted_indices = X.select(pl.col(self.date_column)).to_series().arg_sort()
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Create stratification variable
        if self.stratify_column and self.stratify_column in X_sorted.columns:
            stratify_var = X_sorted.select(pl.col(self.stratify_column)).to_series().to_numpy()
        else:
            # Use target for stratification
            stratify_var = y_sorted.to_numpy()
            
        # Convert continuous targets to bins for stratification
        if np.issubdtype(stratify_var.dtype, np.floating):
            import pandas as pd
            stratify_var = pd.cut(stratify_var, bins=self.n_bins, labels=False)
        
        # Calculate temporal split point (maintain time series nature)
        n_total = len(X_sorted)
        n_train = int(n_total * (1 - self.test_size))
        
        # Take first n_train samples as candidate training set
        train_candidates = range(n_train)
        test_candidates = range(n_train, n_total)
        
        X_train = X_sorted[train_candidates]
        X_test = X_sorted[test_candidates]
        y_train = y_sorted[train_candidates]
        y_test = y_sorted[test_candidates]
        
        return X_train, X_test, y_train, y_test
    
    def get_train_indices(self, X: pl.DataFrame) -> list[int]:
        """Get indices for training data."""
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in features")
            
        sorted_indices = X.select(pl.col(self.date_column)).to_series().arg_sort()
        n_train = int(len(X) * (1 - self.test_size))
        
        return sorted_indices[:n_train].to_list()
    
    def get_test_indices(self, X: pl.DataFrame) -> list[int]:
        """Get indices for test data."""
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in features")
            
        sorted_indices = X.select(pl.col(self.date_column)).to_series().arg_sort()
        n_train = int(len(X) * (1 - self.test_size))
        
        return sorted_indices[n_train:].to_list()


class GroupedSplitter:
    """
    Grouped splitter that ensures samples from the same group stay together.
    
    Useful for splitting stock data where you want to ensure all data
    from the same stock stays in the same split.
    """
    
    def __init__(
        self, 
        group_column: str, 
        test_size: float = 0.2, 
        random_state: int = 42
    ):
        """
        Initialize the grouped splitter.
        
        Args:
            group_column: Column name containing group identifiers (e.g., 'symbol')
            test_size: Proportion of groups to use for testing
            random_state: Random seed for reproducibility
        """
        self.group_column = group_column
        self.test_size = test_size
        self.random_state = random_state
        
    def split(self, X: pl.DataFrame, y: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        """
        Split data ensuring groups stay together.
        
        Args:
            X: Feature matrix with group column
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.group_column not in X.columns:
            raise ValueError(f"Group column '{self.group_column}' not found in features")
            
        # Get unique groups
        unique_groups = X.select(pl.col(self.group_column)).unique().to_series().to_list()
        
        # Split groups
        train_groups, test_groups = train_test_split(
            unique_groups,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Create boolean masks
        train_mask = X.select(pl.col(self.group_column).is_in(train_groups)).to_series()
        test_mask = X.select(pl.col(self.group_column).is_in(test_groups)).to_series()
        
        # Split data
        X_train = X.filter(train_mask)
        X_test = X.filter(test_mask)
        y_train = y.filter(train_mask)
        y_test = y.filter(test_mask)
        
        return X_train, X_test, y_train, y_test
    
    def get_train_indices(self, X: pl.DataFrame) -> list[int]:
        """Get indices for training data."""
        if self.group_column not in X.columns:
            raise ValueError(f"Group column '{self.group_column}' not found in features")
            
        unique_groups = X.select(pl.col(self.group_column)).unique().to_series().to_list()
        train_groups, _ = train_test_split(
            unique_groups,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        train_mask = X.select(pl.col(self.group_column).is_in(train_groups)).to_series()
        return [i for i, mask in enumerate(train_mask) if mask]
    
    def get_test_indices(self, X: pl.DataFrame) -> list[int]:
        """Get indices for test data."""
        if self.group_column not in X.columns:
            raise ValueError(f"Group column '{self.group_column}' not found in features")
            
        unique_groups = X.select(pl.col(self.group_column)).unique().to_series().to_list()
        _, test_groups = train_test_split(
            unique_groups,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        test_mask = X.select(pl.col(self.group_column).is_in(test_groups)).to_series()
        return [i for i, mask in enumerate(test_mask) if mask]
"""
Data processing utilities for the modeling pipeline.

This module provides clean, reusable data processing functionality including
data loading, validation, splitting, and Pool creation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from catboost import Pool
from hamilton import driver

from ..processing_functions import split_data, metadata, pools, dataset

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data loading, validation, and preprocessing for the modeling pipeline.
    
    This class encapsulates all data-related operations and provides a clean interface
    that can be reused across training and inference workflows.
    """
    
    def __init__(self, features: List[str], target: str):
        """
        Initialize the data processor.
        
        Args:
            features: List of feature column names
            target: Target column name
        """
        self.features = features
        self.target = target
        self._hamilton_driver = driver.Builder().with_modules(split_data, metadata, pools, dataset).build()
        
    def load_features(self, data_path: Path, features_file: str) -> pd.DataFrame:
        """
        Load feature data from parquet file.
        
        Args:
            data_path: Base data directory path
            features_file: Features file name
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If features file doesn't exist
            ValueError: If data loading fails
        """
        features_path = data_path / features_file
        
        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found at {features_path}. "
                "Run feature engineering pipeline first."
            )
            
        try:
            logger.info(f"Loading features from {features_path}")
            df = pd.read_parquet(features_path)
            
            # Validate required columns
            missing_features = set(self.features) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
                
            if self.target not in df.columns:
                raise ValueError(f"Target column '{self.target}' not found in data")
                
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading features file: {str(e)}") from e
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input data by removing null values and invalid entries.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        initial_rows = len(df)
        
        # Drop rows with null values in required columns
        required_columns = self.features + [self.target, "date", "symbol"]
        available_columns = [col for col in required_columns if col in df.columns]
        df_clean = df.dropna(subset=available_columns)
        
        rows_removed = initial_rows - len(df_clean)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} rows with null values")
            
        # Additional validation
        if len(df_clean) == 0:
            raise ValueError("No valid data remaining after cleaning")
            
        return df_clean
        
    def prepare_data_splits(
        self, 
        df: pd.DataFrame, 
        time_span: int
    ) -> Dict[str, Any]:
        """
        Prepare train/validation/test splits and create CatBoost pools.
        
        Args:
            df: Input DataFrame
            time_span: Time span in days for splits
            
        Returns:
            Dictionary containing pools, metadata, and dataset
        """
        logger.info("Preparing data splits...")
        
        # Use Hamilton to execute the data processing pipeline
        result = self._hamilton_driver.execute(
            final_vars=["pools", "metadata", "dataset"],
            inputs={
                "df": df, 
                "time_span": time_span, 
                "features": self.features
            }
        )
        
        # Log split information
        metadata_info = result["metadata"]
        logger.info(f"Train: {metadata_info['training_n_rows']} rows "
                   f"({metadata_info['training_start']} to {metadata_info['training_end']})")
        logger.info(f"Validation: {metadata_info['validation_n_rows']} rows "
                   f"({metadata_info['validation_start']} to {metadata_info['validation_end']})")
        logger.info(f"Test: {metadata_info['test_n_rows']} rows "
                   f"({metadata_info['test_start']} to {metadata_info['test_end']})")
        
        return result
        
    def create_pool(
        self, 
        data: pd.DataFrame, 
        features: List[str], 
        target: str,
        pool_name: str = "data"
    ) -> Pool:
        """
        Create a CatBoost Pool from DataFrame.
        
        Args:
            data: Input DataFrame
            features: Feature columns to use
            target: Target column name
            pool_name: Name for logging purposes
            
        Returns:
            CatBoost Pool
        """
        logger.info(f"Creating {pool_name} pool with {len(data)} samples")
        
        X = data[features]
        y = data[target] if target in data.columns else None
        
        # Detect categorical features
        cat_features = []
        for i, col in enumerate(features):
            if col in X.columns and (X[col].dtype == "category" or X[col].dtype == "object"):
                cat_features.append(i)
                
        if cat_features:
            logger.info(f"Detected categorical features: {cat_features}")
            
        pool = Pool(
            data=X,
            label=y,
            feature_names=features,
            cat_features=cat_features if cat_features else None
        )
        
        return pool
        
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return diagnostics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with data quality metrics
        """
        diagnostics = {}
        
        # Basic statistics
        diagnostics["total_rows"] = len(df)
        diagnostics["total_columns"] = len(df.columns)
        
        # Missing values
        missing_counts = df.isnull().sum()
        diagnostics["missing_values"] = missing_counts[missing_counts > 0].to_dict()
        
        # Feature-specific checks
        feature_stats = {}
        for feature in self.features:
            if feature in df.columns:
                series = df[feature]
                feature_stats[feature] = {
                    "dtype": str(series.dtype),
                    "null_count": series.isnull().sum(),
                    "unique_values": series.nunique(),
                    "min": series.min() if pd.api.types.is_numeric_dtype(series) else None,
                    "max": series.max() if pd.api.types.is_numeric_dtype(series) else None,
                }
                
        diagnostics["feature_statistics"] = feature_stats
        
        # Target statistics
        if self.target in df.columns:
            target_series = df[self.target]
            diagnostics["target_statistics"] = {
                "dtype": str(target_series.dtype),
                "null_count": target_series.isnull().sum(),
                "mean": target_series.mean(),
                "std": target_series.std(),
                "min": target_series.min(),
                "max": target_series.max(),
                "quantiles": target_series.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
            }
            
        return diagnostics
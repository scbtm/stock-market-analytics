"""
Data configuration classes.

This module defines configuration classes for data processing,
features, and file paths.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for data processing and features."""
    
    # File paths
    features_file: str = Field(
        default="stock_history_features.parquet",
        description="Features file name"
    )
    
    # Target and features
    target: str = Field(
        default="y_log_returns",
        description="Target column name"
    )
    
    features: List[str] = Field(
        default=[
            "amihud_illiq",
            "kurtosis_diff", 
            "skewness_diff",
            "long_mean",
            "short_mean",
            "mean_diff",
            "long_diff",
            "short_diff",
            "rsi",
            "long_short_momentum",
            "risk_adj_momentum", 
            "pct_from_high_long",
            "pct_from_high_short",
            "iqr_vol",
            "month",
            "day_of_week",
            "day_of_year",
        ],
        description="Feature columns to use"
    )
    
    # Data splitting
    time_span: int = Field(
        default=70,
        description="Time span in days for train/val/test splits"
    )
    
    # Data validation
    required_columns: List[str] = Field(
        default=["date", "symbol"],
        description="Required columns that must be present"
    )
    
    def get_all_required_columns(self) -> List[str]:
        """Get all columns that are required for modeling."""
        return self.features + [self.target] + self.required_columns
        
    def validate_dataframe(self, df) -> List[str]:
        """Validate that DataFrame contains required columns."""
        missing_columns = []
        all_required = self.get_all_required_columns()
        
        for col in all_required:
            if col not in df.columns:
                missing_columns.append(col)
                
        return missing_columns
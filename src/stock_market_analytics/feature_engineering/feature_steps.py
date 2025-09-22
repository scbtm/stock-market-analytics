"""
Simple functions that coordinate core feature engineering components and can be
reused across different flows and scenarios.
"""

from typing import Any

import pandas as pd
import polars as pl
from hamilton import driver

from stock_market_analytics.config import config
from stock_market_analytics.feature_engineering import feature_pipeline


def load_stock_data(
    stocks_history_path: str,
) -> pl.DataFrame:
    """Load and validate stock data from Parquet file."""

    try:
        return pl.read_parquet(stocks_history_path)
    except Exception as e:
        raise ValueError(f"Error loading stocks history file: {str(e)}") from e


def apply_time_filters(data: pl.DataFrame, past_horizon: int) -> pl.DataFrame:
    """Apply time-based filters to limit data lookback."""
    if past_horizon > 0:
        max_date = data["date"].max()
        if max_date is not None:
            # Convert to pandas timestamp for subtraction
            max_date_pd = pd.Timestamp(max_date)
            max_lookback_date = max_date_pd - pd.Timedelta(days=past_horizon)
            return data.filter(pl.col("date") >= pl.lit(max_lookback_date))
    return data


def create_feature_pipeline() -> driver.Driver:
    """Create Hamilton feature pipeline driver."""
    return driver.Builder().with_modules(feature_pipeline).build()


def execute_feature_pipeline(
    dr: driver.Driver, raw_data: pl.DataFrame, config_dict: dict[str, Any]
) -> pl.DataFrame:
    """Execute feature pipeline with input data and configuration."""
    results = dr.execute(
        final_vars=["df_features"],
        inputs={"raw_df": raw_data, **config_dict},
    )
    return results["df_features"]


def save_features(
    data: pl.DataFrame,
    features_path: str,
) -> None:
    """Save engineered features to parquet file."""
    data.write_parquet(features_path)


def build_features_from_data(
    stocks_history_path: str = "stocks_history.parquet",
    features_path: str = "stock_history_features.parquet",
) -> dict[str, Any]:
    """Complete feature building workflow from data path."""
    # Load data
    data = load_stock_data(stocks_history_path)

    # Apply time filters
    filtered_data = apply_time_filters(data, config.feature_engineering.past_horizon)

    # Create and execute pipeline
    pipeline = create_feature_pipeline()
    features = execute_feature_pipeline(
        pipeline, filtered_data, config.feature_engineering.as_dict
    )

    # Save features
    save_features(features, features_path)

    return {
        "status": "success",
        "input_records": len(filtered_data),
        "output_records": len(features),
        "features_file": features_path,
    }

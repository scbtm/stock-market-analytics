"""
Online Inference Functions for Stock Market Analytics

This module provides simple functions for real-time inference on individual stock symbols.
It leverages the existing data collection and feature engineering infrastructure with minimal overhead.

Key principles:
1. Reuse existing collection_steps.collect_and_process_symbol for data collection
2. Reuse existing feature_steps.execute_feature_pipeline for feature generation
3. Never filter out null targets (needed for inference)
4. Collect sufficient historical data based on past_horizon configuration
"""

import polars as pl

from stock_market_analytics.data_collection.collection_steps import (
    collect_and_process_symbol,
)
from stock_market_analytics.feature_engineering import features_config
from stock_market_analytics.feature_engineering.feature_steps import (
    create_feature_pipeline,
    execute_feature_pipeline,
)


def create_inference_collection_plan(symbol: str) -> dict[str, str]:
    """
    Create a collection plan optimized for inference data requirements.

    For inference, we need enough historical data to generate all features.
    Based on config, the longest lookback is past_horizon (1960 days = ~5.4 years).
    We use "max" period to ensure we have sufficient data for all features.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Collection plan dictionary ready for collect_and_process_symbol
    """
    return {
        "symbol": symbol.upper(),
        "period": "1y",  # Use 1 year to limit data volume while ensuring enough history
    }


def collect_inference_data(symbol: str) -> pl.DataFrame:
    """
    Collect and process data for inference using existing collection infrastructure.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Processed and validated stock data ready for feature engineering

    Raises:
        RuntimeError: If data collection fails
    """
    print(f"📊 Collecting inference data for {symbol}...")

    # Create collection plan
    collection_plan = create_inference_collection_plan(symbol)

    # Use existing collection infrastructure
    result = collect_and_process_symbol(collection_plan)

    # Check collection success
    if result["data"] is None:
        metadata = result.get("new_metadata", {})
        status = metadata.get("status", "unknown_error")
        raise RuntimeError(f"Data collection failed for {symbol}. Status: {status}")

    data = result["data"]
    print(f"✅ Collected {len(data)} data points for {symbol}")
    print(f"📈 Date range: {data['date'].min()} to {data['date'].max()}")

    return data


def generate_inference_features(raw_data: pl.DataFrame) -> pl.DataFrame:
    """
    Generate features for inference using existing feature engineering pipeline.

    Unlike training, we do NOT filter out rows with null targets since we need
    the latest data points for inference even if they don't have future returns.

    Args:
        raw_data: Raw stock data from collect_inference_data

    Returns:
        DataFrame with engineered features ready for model prediction

    Raises:
        RuntimeError: If feature generation fails
    """
    print("🧮 Generating features for inference...")

    try:
        # Create Hamilton pipeline
        pipeline = create_feature_pipeline()

        # Execute feature pipeline with configuration
        # Note: We do NOT apply time filters here since we want all available data
        # for inference, especially the most recent points
        features = execute_feature_pipeline(pipeline, raw_data, features_config.as_dict)

        print(
            f"✅ Generated {features.shape[1]} features from {features.shape[0]} data points"
        )
        print(
            f"📅 Feature date range: {features['date'].min()} to {features['date'].max()}"
        )

        return features

    except Exception as e:
        raise RuntimeError(f"Feature generation failed: {str(e)}") from e

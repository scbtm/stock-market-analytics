"""
Inference Module for Stock Market Analytics

This module provides clean, simple online inference capabilities for individual stock symbols,
leveraging the existing data collection and feature engineering infrastructure.

Main Functions:
- get_inference_data: Complete pipeline for inference-ready features
- get_latest_features: Get the most recent feature data point
- collect_inference_data: Collect and process raw data using existing infrastructure
- generate_inference_features: Generate features using existing Hamilton pipeline

Convenience Aliases:
- infer: Shorthand for get_latest_features
- collect: Shorthand for get_inference_data

Quick Usage:
    from stock_market_analytics.inference import infer, collect

    # Get latest features for prediction
    latest = infer("AAPL")

    # Get full feature dataset
    full_data = collect("MSFT")
"""

from .inference_functions import (
    collect_inference_data,
    create_inference_collection_plan,
    generate_inference_features,
)

__all__ = [
    "create_inference_collection_plan",
    "collect_inference_data",
    "generate_inference_features",
]

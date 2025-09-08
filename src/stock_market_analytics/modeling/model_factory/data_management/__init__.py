"""
Data management components for stock market analytics modeling.

This module provides data splitting strategies, preprocessing utilities,
and other data management tools.
"""

from .splitters import (
    TimeSeriesDataSplitter,
)

__all__ = [
    "TimeSeriesDataSplitter",
]
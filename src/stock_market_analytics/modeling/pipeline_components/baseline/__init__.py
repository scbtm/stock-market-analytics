"""
Baseline components for stock market analytics modeling.

This module provides baseline predictors and baseline functions.
"""

from .naive_baselines import (
    HistoricalQuantileBaseline,
)

__all__ = [
    "HistoricalQuantileBaseline",
]
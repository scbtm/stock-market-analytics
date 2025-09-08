from __future__ import annotations

"""
Simplified protocols for stock market modeling pipeline.

Core philosophy: Keep it simple and focused on actual use cases.
- Basic model interfaces for experimentation
- Simple evaluation structure  
- Minimal abstractions that help organize code
"""

from typing import Any, Literal, Mapping, Protocol, Sequence
import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]

# =========================
# Core model interfaces
# =========================

@Protocol
class QuantilePredictor(Protocol):
    """Model that can predict quantiles - our main use case."""
    def fit(self, X: Any, y: Any) -> "QuantilePredictor": ...
    def predict_quantiles(self, X: Any, quantiles: Sequence[float]) -> NDArrayF: ...

# =========================
# Simple evaluation protocol
# =========================

@Protocol  
class Evaluator(Protocol):
    """Simple evaluator for quantile regression models."""
    def evaluate(self, y_true: NDArrayF, y_pred: NDArrayF, quantiles: Sequence[float]) -> dict[str, float]: ...


# =========================
# Data splitting - simplified for time series
# =========================

@Protocol
class DataSplitter(Protocol):
    """Simple data splitter for time series experiments."""
    def train_test_split(self, data: Any, test_size: float = 0.2) -> tuple[Any, Any]: ...
"""
Simplified pipeline components for stock market analytics.

Focus on practical experimentation tools rather than over-abstracted protocols.
"""

# Core simplified protocols
from .protocols import (
    QuantilePredictor,
    Evaluator, 
    DataSplitter,
    NDArrayF,
)

# Utilities  
from .prediction.prediction_functions import validate_quantiles
from .evaluation.evaluation_functions import compute_quantile_loss

# Import existing implementations (these will need simplification too)
try:
    from .prediction import (
        CatBoostMultiQuantileModel, 
        HistoricalQuantileBaseline,
    )
except ImportError:
    pass

try:
    from .evaluation import (
        QuantileRegressionEvaluator,
    )
except ImportError:
    pass

try:
    from .calibration import (
        ConformalCalibrator,
    )
except ImportError:
    pass

__all__ = [
    # Core protocols
    "QuantilePredictor",
    "Evaluator", 
    "DataSplitter",
    "NDArrayF",
    
    # Utilities
    "validate_quantiles",
    "compute_quantile_loss",
    
    # Models (if available)
    "CatBoostMultiQuantileModel",
    "HistoricalQuantileBaseline", 
    "QuantileRegressionEvaluator",
    "ConformalCalibrator",
]
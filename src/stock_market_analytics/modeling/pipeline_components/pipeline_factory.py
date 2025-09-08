"""
Protocol-based pipeline factory for creating consistent ML pipelines.

This module provides a simplified pipeline factory that uses the protocol-compliant
components for flexible model experimentation.
"""

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_market_analytics.config import config
from .factories import QuantileRegressionModelFactory

# Use the protocol-compliant model factory
model_factory = QuantileRegressionModelFactory()

# Feature preprocessing pipeline
def create_preprocessing_pipeline():
    """Create a standardized preprocessing pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=1)),  # Retain 95% variance
    ])


def get_pipeline(model_type: str = "catboost") -> Pipeline:
    """
    Create a complete ML pipeline with preprocessing and model.
    
    Args:
        model_type: Type of model to use ("catboost", "historical")
        
    Returns:
        Complete sklearn Pipeline
    """
    # Create preprocessing pipeline
    preprocessing = create_preprocessing_pipeline()
    
    # Create model using factory
    if model_type == "catboost":
        model = model_factory.create_model(
            "catboost", 
            **config.modeling.cb_model_params
        )
    elif model_type == "historical":
        model = model_factory.create_model("historical")
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available options: {model_factory.get_available_models()}"
        )
    
    # Create complete pipeline
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])


def get_baseline_pipeline(baseline_type: str = "historical") -> Pipeline:
    """
    Create a baseline pipeline for comparison.
    
    Args:
        baseline_type: Type of baseline ("historical")
        
    Returns:
        Baseline pipeline
    """
    return get_pipeline(baseline_type)


# For backward compatibility with existing code
pipeline = get_pipeline("catboost")
baseline_pipelines = {
    "historical": get_pipeline("historical"),
}
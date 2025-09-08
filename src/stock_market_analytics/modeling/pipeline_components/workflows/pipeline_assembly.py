"""
Complete pipeline assembly for stock market analytics modeling.

This module provides high-level functions for creating complete ML pipelines
with preprocessing and models.
"""

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_market_analytics.config import config
from stock_market_analytics.modeling.pipeline_components.prediction.factories import QuantileRegressionModelFactory

# Use the protocol-compliant model factory
model_factory = QuantileRegressionModelFactory()


def create_preprocessing_pipeline():
    """Create a standardized preprocessing pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=1)),  # Retain 95% variance
    ])


def get_pipeline(model_type: str = "catboost", **model_params) -> Pipeline:
    """
    Create a complete ML pipeline with preprocessing and model.
    
    Args:
        model_type: Type of model to use (e.g., "catboost", "historical")
        **model_params: Additional parameters to pass to model creation
        
    Returns:
        Complete sklearn Pipeline
    """
    # Create preprocessing pipeline
    preprocessing = create_preprocessing_pipeline()
    
    # Create model using factory (it handles all the logic)
    model = model_factory.create_model(model_type, **model_params)
    
    # Create complete pipeline
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])





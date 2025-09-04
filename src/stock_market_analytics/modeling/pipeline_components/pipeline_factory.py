from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from stock_market_analytics.modeling.pipeline_components.configs import (
    modeling_config as modeling_config,
)
from stock_market_analytics.modeling.pipeline_components.parameters import (
    cb_model_params,
    pca_params,
)
from stock_market_analytics.modeling.pipeline_components.predictors import (
    CatBoostMultiQuantileModel,
)

QUANTILES = modeling_config["QUANTILES"]

quantile_regressor = CatBoostMultiQuantileModel(
            quantiles=QUANTILES,
            **cb_model_params
        )

pca = PCA(**pca_params)

pipeline = Pipeline(steps=[
    ("pca", pca),
    ("quantile_regressor", quantile_regressor)
])

# This function gets modified according to the intended pipeline to be used. The one defined above is an example.
def get_pipeline() -> Pipeline:
    """
    Returns the machine learning pipeline for stock market analytics.

    The pipeline consists of:
    1. PCA for dimensionality reduction.
    2. CatBoost MultiQuantile Regressor for quantile regression.

    Returns:
        Pipeline: A scikit-learn Pipeline object.
    """
    return pipeline

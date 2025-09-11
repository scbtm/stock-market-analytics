"""
Ad-hoc modeling functions for specific tasks.

This module contains specialized functions for performing specific modeling tasks,
designed to be modified based on experimental needs. Functions here are expected
to change as experiments evolve.

These are pure business logic functions with no Metaflow dependencies,
following the steps architecture pattern.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from stock_market_analytics.config import config

from stock_market_analytics.modeling.model_factory.data_management.preprocessing import (
    get_modeling_sets,
)

from stock_market_analytics.modeling.model_factory.estimation.estimators import (
    CatBoostMultiQuantileModel,
)

from stock_market_analytics.modeling.model_factory.calibration.calibrators import (
    QuantileConformalCalibrator,
)

from stock_market_analytics.modeling.model_factory.evaluation.evaluators import (
    QuantileRegressionEvaluator,
)


def load_features_data(data_path: str) -> pd.DataFrame:
    """
    Load the features dataset for modeling.

    Args:
        data_path: Path to data directory

    Returns:
        Features DataFrame

    Raises:
        FileNotFoundError: If features file not found
        ValueError: If data is invalid
    """
    features_file = Path(data_path) / config.modeling.features_file

    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")

    try:
        # load features data
        df = pd.read_parquet(features_file)

        #drop nulls in target column
        df = df.dropna(subset=[config.modeling.target])

        if df.empty:
            raise ValueError("Features file is empty")

        return df

    except Exception as e:
        raise ValueError(f"Error loading features file: {str(e)}") from e


def prepare_modeling_data(
    df: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Prepare data for modeling by selecting features and target.

    Args:
        df: Raw features DataFrame

    Returns:
        Dictionary with keys 'features' and 'target'
    """
    # Get modeling sets
    modeling_sets = get_modeling_sets(df, date_col='date', symbol_col='symbol', feature_cols=config.modeling.features, target_col=config.modeling.target)

    return modeling_sets

def get_adhoc_transforms_pipeline() -> Pipeline:
    """
    Create a pipeline of ad-hoc data transformations.

    Returns:
        Pipeline with transformations
    """
    # ------------------------------- PCA for feature groups -------------------------------
    feature_groups = config.modeling.feature_groups

    pca_groups = {group.lower() + "_pca": PCA(**config.modeling.pca_group_params[group]) for group in feature_groups}

    scaler_groups = {group.lower() + "_scaler": Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")),("scaler", StandardScaler()),]) for group in feature_groups}

    # ------------------------------- Transformation Pipeline -------------------------------
    _scalers = [(name, scaler, feature_groups[group]) for group, name, scaler in zip(feature_groups, scaler_groups.keys(), scaler_groups.values(), strict=True)]

    scalers = ColumnTransformer(transformers=_scalers,
                                remainder="drop",  # Drop any features not specified in the transformers
                                verbose_feature_names_out="{feature_name}__scaled",
                                verbose=False,
                                ).set_output(transform="pandas")

    _reducers = [(name, pca, [f"{feature_name}__scaled" for feature_name in feature_groups[group]]) for group, name, pca in zip(feature_groups, pca_groups.keys(), pca_groups.values(), strict=True)]
    reducers = ColumnTransformer(transformers=_reducers,
                                remainder="drop",  # Drop any features not specified in the transformers
                                verbose_feature_names_out=True,
                                verbose=False,
                                ).set_output(transform="pandas")

    transformation_pipeline = Pipeline(steps=[("scalers", scalers), ("reducers", reducers)])

    return transformation_pipeline


def get_catboost_multiquantile_model(params:Optional[dict] = None) -> CatBoostMultiQuantileModel:
    """
    Create a CatBoost multi-quantile regression model.

    Returns:
        Configured CatBoostMultiQuantileModel instance
    """
    if not params:
        params = config.modeling.cb_model_params

    return CatBoostMultiQuantileModel(**params)


def get_calibrator() -> QuantileConformalCalibrator:

    # Initialize conformal calibrator for 80% prediction intervals
    alpha = 0.2  # 80% coverage
    conformal_calibrator = QuantileConformalCalibrator(
        alpha=alpha,
        method="absolute"  # Could also use "normalized" with uncertainty estimates
        )

    return conformal_calibrator

def get_evaluator() -> QuantileRegressionEvaluator:

    quantiles = config.modeling.quantiles

    return QuantileRegressionEvaluator(quantiles=quantiles)

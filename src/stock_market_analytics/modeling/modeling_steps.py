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
        Dictionary with keys for each data split containing (X, y) tuples
        
    Raises:
        ValueError: If required columns are missing or data splits fail
    """
    try:
        # Validate required columns exist
        required_cols = set(config.modeling.features + [config.modeling.target, 'date', 'symbol'])
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Get modeling sets
        modeling_sets = get_modeling_sets(
            df, 
            date_col='date', 
            symbol_col='symbol', 
            feature_cols=config.modeling.features, 
            target_col=config.modeling.target
        )
        
        # Validate that all splits have data
        expected_splits = ['train', 'val', 'cal', 'test']
        for split_name in expected_splits:
            if split_name not in modeling_sets:
                raise ValueError(f"Missing data split: {split_name}")
            X, y = modeling_sets[split_name]
            if X.empty or y.empty:
                raise ValueError(f"Empty data in {split_name} split")

        return modeling_sets
        
    except Exception as e:
        raise ValueError(f"Error preparing modeling data: {str(e)}") from e

def get_adhoc_transforms_pipeline() -> Pipeline:
    """
    Create a pipeline of ad-hoc data transformations.

    Returns:
        Pipeline with transformations
        
    Raises:
        ValueError: If feature groups configuration is invalid
        KeyError: If PCA parameters are missing for feature groups
    """
    try:
        # ------------------------------- PCA for feature groups -------------------------------
        feature_groups = config.modeling.feature_groups
        
        if not feature_groups:
            raise ValueError("No feature groups defined in configuration")

        # Validate PCA parameters exist for all groups
        missing_pca_params = set(feature_groups.keys()) - set(config.modeling.pca_group_params.keys())
        if missing_pca_params:
            raise KeyError(f"Missing PCA parameters for groups: {missing_pca_params}")

        pca_groups = {
            group.lower() + "_pca": PCA(**config.modeling.pca_group_params[group]) 
            for group in feature_groups
        }

        scaler_groups = {
            group.lower() + "_scaler": Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]) 
            for group in feature_groups
        }

        # ------------------------------- Transformation Pipeline -------------------------------
        _scalers = [
            (name, scaler, feature_groups[group]) 
            for group, name, scaler in zip(
                feature_groups, scaler_groups.keys(), scaler_groups.values(), strict=True
            )
        ]

        scalers = ColumnTransformer(
            transformers=_scalers,
            remainder="drop",  # Drop any features not specified in the transformers
            verbose_feature_names_out="{feature_name}__scaled",
            verbose=False,
        ).set_output(transform="pandas")

        _reducers = [
            (name, pca, [f"{feature_name}__scaled" for feature_name in feature_groups[group]]) 
            for group, name, pca in zip(
                feature_groups, pca_groups.keys(), pca_groups.values(), strict=True
            )
        ]
        
        reducers = ColumnTransformer(
            transformers=_reducers,
            remainder="drop",  # Drop any features not specified in the transformers
            verbose_feature_names_out=True,
            verbose=False,
        ).set_output(transform="pandas")

        transformation_pipeline = Pipeline(steps=[("scalers", scalers), ("reducers", reducers)])

        return transformation_pipeline
        
    except Exception as e:
        raise ValueError(f"Error creating transformation pipeline: {str(e)}") from e


def get_catboost_multiquantile_model(params: Optional[dict] = None) -> CatBoostMultiQuantileModel:
    """
    Create a CatBoost multi-quantile regression model.

    Args:
        params: Optional model parameters. If None, uses config defaults.

    Returns:
        Configured CatBoostMultiQuantileModel instance
        
    Raises:
        ValueError: If model parameters are invalid
    """
    try:
        if not params:
            params = config.modeling.cb_model_params
            
        # Validate quantiles parameter if present
        if 'quantiles' in params:
            quantiles = params['quantiles']
            if not isinstance(quantiles, (list, tuple)) or len(quantiles) == 0:
                raise ValueError("Quantiles must be a non-empty list or tuple")
            if any(q <= 0 or q >= 1 for q in quantiles):
                raise ValueError("All quantiles must be between 0 and 1 (exclusive)")

        return CatBoostMultiQuantileModel(**params)
        
    except Exception as e:
        raise ValueError(f"Error creating CatBoost model: {str(e)}") from e


def get_calibrator() -> QuantileConformalCalibrator:
    """
    Create a quantile conformal calibrator.
    
    Returns:
        Configured QuantileConformalCalibrator instance
        
    Raises:
        ValueError: If calibrator configuration is invalid
    """
    try:
        # Initialize conformal calibrator for 80% prediction intervals
        alpha = 0.2  # 80% coverage
        
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
            
        conformal_calibrator = QuantileConformalCalibrator(
            alpha=alpha,
            method="absolute"  # Could also use "normalized" with uncertainty estimates
        )

        return conformal_calibrator
        
    except Exception as e:
        raise ValueError(f"Error creating conformal calibrator: {str(e)}") from e

def get_evaluator() -> QuantileRegressionEvaluator:
    """
    Create a quantile regression evaluator.
    
    Returns:
        Configured QuantileRegressionEvaluator instance
        
    Raises:
        ValueError: If evaluator configuration is invalid
    """
    try:
        quantiles = config.modeling.quantiles
        
        if not quantiles or not isinstance(quantiles, (list, tuple)):
            raise ValueError("Quantiles must be a non-empty list or tuple")
            
        if any(q <= 0 or q >= 1 for q in quantiles):
            raise ValueError("All quantiles must be between 0 and 1 (exclusive)")

        return QuantileRegressionEvaluator(quantiles=quantiles)
        
    except Exception as e:
        raise ValueError(f"Error creating quantile regression evaluator: {str(e)}") from e

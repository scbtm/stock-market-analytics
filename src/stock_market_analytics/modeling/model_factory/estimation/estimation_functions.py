"""
Helper functions for ML estimators.

This module provides utility functions and preprocessing helpers
for various machine learning estimators and model implementations.
"""

import numpy as np
import pandas as pd
from typing import Any, Sequence
from catboost import Pool


def detect_categorical_features(
    X: pd.DataFrame,
) -> Sequence[str | int]:
    """
    Detect categorical features in a Pandas DataFrame.

    Args:
        X: Feature matrix
    Returns:
        List of categorical feature names
    """
    if isinstance(X, pd.DataFrame):  # Pandas DataFrame
        categorical_cols = X.select_dtypes(
            include=["category", "object"]
        ).columns.tolist()
        return categorical_cols
    else:
        return []


def create_catboost_pool(
    X: pd.DataFrame,
    y: pd.Series | None = None,
) -> Any:
    """
    Create a CatBoost Pool object from Polars DataFrame.

    Args:
        X: Feature matrix
        y: Target vector (optional for prediction)
        categorical_features: List of categorical feature names

    Returns:
        CatBoost Pool object
    """
    # Identify categorical feature indices
    cat_features = detect_categorical_features(X)
    return Pool(data=X, label=y, cat_features=cat_features if cat_features else None)


# def prepare_features_for_sklearn(
#     X: pl.DataFrame,
#     feature_names: List[str] | None = None
# ) -> Tuple[np.ndarray, List[str]]:
#     """
#     Prepare features for sklearn-compatible estimators.

#     Args:
#         X: Feature matrix
#         feature_names: Specific features to use (None for all)

#     Returns:
#         Tuple of (feature_array, feature_names)
#     """
#     if feature_names is None:
#         feature_names = X.columns

#     # Select only specified features
#     selected_features = [col for col in feature_names if col in X.columns]
#     X_selected = X.select(selected_features)

#     # Convert to numpy
#     X_array = X_selected.to_numpy()

#     return X_array, selected_features


# def handle_missing_values(
#     X: pl.DataFrame,
#     strategy: str = "median",
#     fill_value: float | None = None
# ) -> pl.DataFrame:
#     """
#     Handle missing values in feature matrix.

#     Args:
#         X: Feature matrix
#         strategy: Strategy for handling missing values ("mean", "median", "mode", "constant")
#         fill_value: Value to use for "constant" strategy

#     Returns:
#         Feature matrix with missing values handled
#     """
#     if strategy == "mean":
#         return X.fill_null(X.mean())
#     elif strategy == "median":
#         return X.fill_null(X.median())
#     elif strategy == "mode":
#         return X.fill_null(X.mode())
#     elif strategy == "constant":
#         if fill_value is None:
#             fill_value = 0.0
#         return X.fill_null(fill_value)
#     else:
#         raise ValueError(f"Unknown strategy: {strategy}")


# def create_polynomial_features(
#     X: pl.DataFrame,
#     degree: int = 2,
#     include_bias: bool = False,
#     feature_subset: List[str] | None = None
# ) -> pl.DataFrame:
#     """
#     Create polynomial features from existing features.

#     Args:
#         X: Feature matrix
#         degree: Polynomial degree
#         include_bias: Whether to include bias term
#         feature_subset: Subset of features to use (None for all numeric features)

#     Returns:
#         Extended feature matrix with polynomial features
#     """
#     from sklearn.preprocessing import PolynomialFeatures

#     # Select numeric features
#     if feature_subset is None:
#         numeric_cols = X.select(pl.col(pl.NUMERIC_DTYPES)).columns
#     else:
#         numeric_cols = [col for col in feature_subset if col in X.columns]

#     if not numeric_cols:
#         return X

#     # Extract numeric data
#     X_numeric = X.select(numeric_cols).to_numpy()

#     # Create polynomial features
#     poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
#     X_poly = poly.fit_transform(X_numeric)

#     # Create new column names
#     poly_feature_names = poly.get_feature_names_out(numeric_cols)

#     # Convert back to Polars DataFrame
#     X_poly_df = pl.DataFrame(X_poly, schema=poly_feature_names)

#     # Combine with non-numeric features
#     non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
#     if non_numeric_cols:
#         X_non_numeric = X.select(non_numeric_cols)
#         result = pl.concat([X_non_numeric, X_poly_df], how="horizontal")
#     else:
#         result = X_poly_df

#     return result


# def feature_importance_to_dict(
#     feature_names: List[str],
#     importance_values: np.ndarray
# ) -> Dict[str, float]:
#     """
#     Convert feature importance arrays to dictionaries.

#     Args:
#         feature_names: List of feature names
#         importance_values: Array of importance values

#     Returns:
#         Dictionary mapping feature names to importance values
#     """
#     return dict(zip(feature_names, importance_values.astype(float)))


# def scale_features(
#     X: pl.DataFrame,
#     method: str = "standard",
#     feature_subset: List[str] | None = None
# ) -> Tuple[pl.DataFrame, Any]:
#     """
#     Scale features using sklearn scalers.

#     Args:
#         X: Feature matrix
#         method: Scaling method ("standard", "minmax", "robust")
#         feature_subset: Features to scale (None for all numeric features)

#     Returns:
#         Tuple of (scaled_features, fitted_scaler)
#     """
#     from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

#     # Select features to scale
#     if feature_subset is None:
#         numeric_cols = X.select(pl.col(pl.NUMERIC_DTYPES)).columns
#     else:
#         numeric_cols = [col for col in feature_subset if col in X.columns]

#     if not numeric_cols:
#         return X, None

#     # Choose scaler
#     if method == "standard":
#         scaler = StandardScaler()
#     elif method == "minmax":
#         scaler = MinMaxScaler()
#     elif method == "robust":
#         scaler = RobustScaler()
#     else:
#         raise ValueError(f"Unknown scaling method: {method}")

#     # Scale numeric features
#     X_numeric = X.select(numeric_cols)
#     X_scaled_array = scaler.fit_transform(X_numeric.to_numpy())
#     X_scaled_df = pl.DataFrame(X_scaled_array, schema=numeric_cols)

#     # Combine with non-numeric features
#     non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
#     if non_numeric_cols:
#         X_non_numeric = X.select(non_numeric_cols)
#         result = pl.concat([X_non_numeric, X_scaled_df], how="horizontal")
#     else:
#         result = X_scaled_df

#     return result, scaler


# def create_interaction_features(
#     X: pl.DataFrame,
#     interaction_pairs: List[Tuple[str, str]]
# ) -> pl.DataFrame:
#     """
#     Create interaction features between specified pairs.

#     Args:
#         X: Feature matrix
#         interaction_pairs: List of (feature1, feature2) pairs

#     Returns:
#         Feature matrix with interaction features added
#     """
#     result = X.clone()

#     for feat1, feat2 in interaction_pairs:
#         if feat1 in X.columns and feat2 in X.columns:
#             interaction_name = f"{feat1}_x_{feat2}"
#             result = result.with_columns(
#                 (pl.col(feat1) * pl.col(feat2)).alias(interaction_name)
#             )

#     return result


# def create_lag_features(
#     X: pl.DataFrame,
#     feature_cols: List[str],
#     lags: List[int],
#     group_col: str | None = None
# ) -> pl.DataFrame:
#     """
#     Create lagged features for time series data.

#     Args:
#         X: Feature matrix with time series data
#         feature_cols: Features to create lags for
#         lags: List of lag periods
#         group_col: Column to group by (e.g., 'symbol' for multiple time series)

#     Returns:
#         Feature matrix with lagged features
#     """
#     result = X.clone()

#     for feature in feature_cols:
#         if feature not in X.columns:
#             continue

#         for lag in lags:
#             lag_name = f"{feature}_lag_{lag}"

#             if group_col is not None:
#                 # Group-wise lagging
#                 result = result.with_columns(
#                     pl.col(feature).shift(lag).over(group_col).alias(lag_name)
#                 )
#             else:
#                 # Simple lagging
#                 result = result.with_columns(
#                     pl.col(feature).shift(lag).alias(lag_name)
#                 )

#     return result


# def create_rolling_features(
#     X: pl.DataFrame,
#     feature_cols: List[str],
#     windows: List[int],
#     operations: List[str] = ["mean", "std"],
#     group_col: str | None = None
# ) -> pl.DataFrame:
#     """
#     Create rolling window features.

#     Args:
#         X: Feature matrix
#         feature_cols: Features to create rolling features for
#         windows: List of window sizes
#         operations: List of operations ("mean", "std", "min", "max", "median")
#         group_col: Column to group by

#     Returns:
#         Feature matrix with rolling features
#     """
#     result = X.clone()

#     for feature in feature_cols:
#         if feature not in X.columns:
#             continue

#         for window in windows:
#             for operation in operations:
#                 rolling_name = f"{feature}_rolling_{window}_{operation}"

#                 if group_col is not None:
#                     # Group-wise rolling
#                     if operation == "mean":
#                         expr = pl.col(feature).rolling_mean(window).over(group_col)
#                     elif operation == "std":
#                         expr = pl.col(feature).rolling_std(window).over(group_col)
#                     elif operation == "min":
#                         expr = pl.col(feature).rolling_min(window).over(group_col)
#                     elif operation == "max":
#                         expr = pl.col(feature).rolling_max(window).over(group_col)
#                     elif operation == "median":
#                         expr = pl.col(feature).rolling_median(window).over(group_col)
#                     else:
#                         continue
#                 else:
#                     # Simple rolling
#                     if operation == "mean":
#                         expr = pl.col(feature).rolling_mean(window)
#                     elif operation == "std":
#                         expr = pl.col(feature).rolling_std(window)
#                     elif operation == "min":
#                         expr = pl.col(feature).rolling_min(window)
#                     elif operation == "max":
#                         expr = pl.col(feature).rolling_max(window)
#                     elif operation == "median":
#                         expr = pl.col(feature).rolling_median(window)
#                     else:
#                         continue

#                 result = result.with_columns(expr.alias(rolling_name))

#     return result


# def validate_feature_matrix(X: pl.DataFrame) -> Dict[str, Any]:
#     """
#     Validate feature matrix and return diagnostics.

#     Args:
#         X: Feature matrix to validate

#     Returns:
#         Dictionary with validation results
#     """
#     diagnostics = {
#         'n_samples': X.height,
#         'n_features': X.width,
#         'missing_values': {},
#         'infinite_values': {},
#         'constant_features': [],
#         'duplicate_features': [],
#         'feature_types': {},
#     }

#     # Check for missing and infinite values
#     for col in X.columns:
#         null_count = X.select(pl.col(col).is_null().sum()).item()
#         diagnostics['missing_values'][col] = null_count

#         # Check for infinite values in numeric columns
#         if X[col].dtype in pl.NUMERIC_DTYPES:
#             inf_count = X.select(pl.col(col).is_infinite().sum()).item()
#             diagnostics['infinite_values'][col] = inf_count

#             # Check for constant features
#             unique_count = X.select(pl.col(col).n_unique()).item()
#             if unique_count <= 1:
#                 diagnostics['constant_features'].append(col)

#         diagnostics['feature_types'][col] = str(X[col].dtype)

#     # Check for duplicate columns
#     seen_cols = set()
#     for col in X.columns:
#         col_hash = hash(tuple(X[col].to_list()))
#         if col_hash in seen_cols:
#             diagnostics['duplicate_features'].append(col)
#         else:
#             seen_cols.add(col_hash)

#     return diagnostics


# def prepare_quantile_targets(
#     y: pl.Series,
#     quantiles: List[float]
# ) -> pl.DataFrame:
#     """
#     Prepare targets for quantile regression.

#     Args:
#         y: Target series
#         quantiles: List of quantile levels

#     Returns:
#         DataFrame with repeated targets for each quantile
#     """
#     n_samples = len(y)
#     n_quantiles = len(quantiles)

#     # Repeat targets for each quantile
#     y_repeated = np.tile(y.to_numpy(), n_quantiles)

#     # Create quantile indicators
#     quantile_indicators = np.repeat(quantiles, n_samples)

#     # Create sample indices
#     sample_indices = np.tile(np.arange(n_samples), n_quantiles)

#     return pl.DataFrame({
#         'target': y_repeated,
#         'quantile': quantile_indicators,
#         'sample_idx': sample_indices
#     })

"""
Ad-hoc modeling functions for specific tasks.

This module contains specialized functions for performing specific modeling tasks,
designed to be modified based on experimental needs. Functions here are expected
to change as experiments evolve.

These are pure business logic functions with no Metaflow dependencies,
following the steps architecture pattern.
"""

from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

from ..config import config
from .model_factory.estimation.estimators import (
    CatBoostQuantileRegressor,
    BaselineEstimator,
)
from .model_factory.evaluation.evaluators import (
    QuantileRegressionEvaluator,
    FinancialRegressionEvaluator,
    BacktestEvaluator,
)
from .model_factory.data_management.splitters import (
    TimeSeriesSplitter,
    WalkForwardSplitter,
)
from .model_factory.data_management.postprocessors import (
    ReturnConstraintProcessor,
    QuantileConsistencyProcessor,
    CompositePredictionProcessor,
)
from .model_factory.calibration.calibrators import (
    QuantileConformalCalibrator,
)


def load_features_data(data_path: Path) -> pl.DataFrame:
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
    features_file = data_path / config.modeling.features_file
    
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    try:
        df = pl.read_parquet(features_file)
        
        if df.is_empty():
            raise ValueError("Features file is empty")
            
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading features file: {str(e)}") from e


def prepare_modeling_data(
    df: pl.DataFrame,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    time_span_days: Optional[int] = None
) -> Tuple[pl.DataFrame, pl.Series]:
    """
    Prepare data for modeling by selecting features and target.
    
    Args:
        df: Raw features DataFrame
        target_column: Target column name (uses config if None)
        feature_columns: List of feature columns (uses config if None)
        time_span_days: Number of recent days to use (uses config if None)
        
    Returns:
        Tuple of (features_df, target_series)
        
    Raises:
        ValueError: If no valid features or target found
    """
    if target_column is None:
        target_column = config.modeling.target
    
    if feature_columns is None:
        feature_columns = config.modeling.features
    
    if time_span_days is None:
        time_span_days = config.modeling.time_span
    
    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Filter recent data if time_span is specified
    if time_span_days > 0 and "date" in df.columns:
        max_date = df.select(pl.col("date").max()).item()
        cutoff_date = max_date - pl.duration(days=time_span_days)
        df = df.filter(pl.col("date") >= cutoff_date)
    
    # Select features that exist in the DataFrame
    available_features = [col for col in feature_columns if col in df.columns]
    
    if not available_features:
        raise ValueError("No specified features found in DataFrame")
    
    # Add date column if available for time series splitting
    if "date" in df.columns:
        available_features = ["date"] + [f for f in available_features if f != "date"]
    
    # Extract features and target
    X = df.select(available_features)
    y = df.select(pl.col(target_column)).to_series()
    
    # Remove rows with missing target
    valid_mask = ~y.is_null()
    X = X.filter(valid_mask)
    y = y.filter(valid_mask)
    
    if X.is_empty() or len(y) == 0:
        raise ValueError("No valid samples after data preparation")
    
    return X, y


def create_modeling_splits(
    X: pl.DataFrame,
    y: pl.Series,
    test_size: float = 0.2,
    validation_size: float = 0.1
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.Series, pl.Series, pl.Series]:
    """
    Create train/validation/test splits for time series modeling.
    
    Args:
        X: Feature matrix
        y: Target series
        test_size: Fraction for test set
        validation_size: Fraction for validation set
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    test_splitter = TimeSeriesSplitter(test_size=test_size)
    X_temp, X_test, y_temp, y_test = test_splitter.split(X, y)
    
    # Second split: separate validation from remaining training data
    val_splitter = TimeSeriesSplitter(test_size=validation_size / (1 - test_size))
    X_train, X_val, y_train, y_val = val_splitter.split(X_temp, y_temp)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_quantile_model(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_val: Optional[pl.DataFrame] = None,
    y_val: Optional[pl.Series] = None,
    quantiles: Optional[List[float]] = None,
    model_params: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Any]] = None
) -> CatBoostQuantileRegressor:
    """
    Fit a CatBoost quantile regressor.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        quantiles: List of quantiles to predict (uses config if None)
        model_params: CatBoost model parameters (uses config if None)
        fit_params: CatBoost fit parameters (uses config if None)
        
    Returns:
        Fitted CatBoost quantile regressor
    """
    if quantiles is None:
        quantiles = config.modeling.quantiles
    
    if model_params is None:
        model_params = config.modeling.cb_model_params
    
    if fit_params is None:
        fit_params = config.modeling.cb_fit_params
    
    # Initialize model
    model = CatBoostQuantileRegressor(
        quantiles=quantiles,
        **model_params
    )
    
    # Prepare fit arguments
    fit_kwargs = fit_params.copy()
    if X_val is not None and y_val is not None:
        fit_kwargs['eval_set'] = (X_val, y_val)
    
    # Fit model
    model.fit(X_train, y_train, **fit_kwargs)
    
    return model


def evaluate_model_performance(
    model: CatBoostQuantileRegressor,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    quantiles: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a quantile regression model.
    
    Args:
        model: Fitted quantile regression model
        X_test: Test features
        y_test: Test targets
        quantiles: List of quantiles to evaluate (uses config if None)
        
    Returns:
        Dictionary of evaluation results
    """
    if quantiles is None:
        quantiles = config.modeling.quantiles
    
    # Make predictions
    y_pred_quantiles = model.predict(X_test)
    y_pred_point = y_pred_quantiles[:, config.modeling.quantile_indices["MID"]]  # Median
    
    # Convert to numpy for evaluation
    y_true = y_test.to_numpy()
    
    # Quantile evaluation
    quantile_evaluator = QuantileRegressionEvaluator(quantiles)
    quantile_metrics = quantile_evaluator.evaluate_quantiles(
        y_true, y_pred_quantiles, quantiles
    )
    
    # Financial evaluation
    financial_evaluator = FinancialRegressionEvaluator()
    financial_metrics = financial_evaluator.evaluate(y_true, y_pred_point)
    
    # Backtest evaluation
    backtest_evaluator = BacktestEvaluator()
    backtest_metrics = backtest_evaluator.evaluate(y_true, y_pred_point)
    
    # Interval evaluation (90% interval example)
    low_idx = config.modeling.quantile_indices["LOW"]
    high_idx = config.modeling.quantile_indices["HIGH"]
    interval_metrics = quantile_evaluator.evaluate_intervals(
        y_true,
        y_pred_quantiles[:, low_idx],
        y_pred_quantiles[:, high_idx],
        alpha=0.1
    )
    
    return {
        "quantile_metrics": quantile_metrics,
        "financial_metrics": financial_metrics,
        "backtest_metrics": backtest_metrics,
        "interval_metrics": interval_metrics,
        "feature_importance": model.get_feature_importance()
    }


def apply_conformal_calibration(
    base_model: CatBoostQuantileRegressor,
    X_cal: pl.DataFrame,
    y_cal: pl.Series,
    alpha: float = 0.1
) -> QuantileConformalCalibrator:
    """
    Apply conformal calibration to a base model.
    
    Args:
        base_model: Fitted base quantile regression model
        X_cal: Calibration features
        y_cal: Calibration targets
        alpha: Miscoverage level for conformal prediction
        
    Returns:
        Fitted conformal calibrator
    """
    # Get base model predictions on calibration set
    y_pred_cal = base_model.predict(X_cal)
    y_pred_point_cal = y_pred_cal[:, config.modeling.quantile_indices["MID"]]
    
    # Create and fit conformal calibrator
    calibrator = QuantileConformalCalibrator(alpha=alpha)
    calibrator.fit(X_cal, y_cal, y_pred_point_cal)
    
    return calibrator


def create_prediction_postprocessor(
    postprocessing_config: Optional[Dict[str, Any]] = None
) -> CompositePredictionProcessor:
    """
    Create a composite post-processor for predictions.
    
    Args:
        postprocessing_config: Configuration for post-processing
        
    Returns:
        Composite prediction post-processor
    """
    if postprocessing_config is None:
        postprocessing_config = {
            "return_constraints": True,
            "quantile_consistency": True,
        }
    
    processors = []
    
    # Return constraint processor
    if postprocessing_config.get("return_constraints", False):
        processors.append(ReturnConstraintProcessor())
    
    # Quantile consistency processor
    if postprocessing_config.get("quantile_consistency", False):
        processors.append(QuantileConsistencyProcessor(config.modeling.quantiles))
    
    return CompositePredictionProcessor(processors)


def run_walk_forward_validation(
    X: pl.DataFrame,
    y: pl.Series,
    initial_train_size: int,
    test_size: int,
    step_size: int = 1,
    max_folds: int = 10,
    model_params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Run walk-forward validation for time series modeling.
    
    Args:
        X: Full feature matrix
        y: Full target series
        initial_train_size: Size of initial training window
        test_size: Size of test window
        step_size: Step size for walking forward
        max_folds: Maximum number of folds to run
        model_params: Model parameters (uses config if None)
        
    Returns:
        List of evaluation results for each fold
    """
    if model_params is None:
        model_params = config.modeling.cb_model_params
    
    # Create walk-forward splitter
    splitter = WalkForwardSplitter(
        initial_train_size=initial_train_size,
        test_size=test_size,
        step_size=step_size
    )
    
    results = []
    
    for fold, (X_train, X_test, y_train, y_test) in enumerate(splitter.split_generator(X, y)):
        if fold >= max_folds:
            break
        
        # Fit model
        model = CatBoostQuantileRegressor(
            quantiles=config.modeling.quantiles,
            **model_params
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        fold_results = evaluate_model_performance(model, X_test, y_test)
        fold_results["fold"] = fold
        fold_results["train_size"] = len(y_train)
        fold_results["test_size"] = len(y_test)
        
        results.append(fold_results)
    
    return results


def create_baseline_models(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.Series
) -> Dict[str, Dict[str, Any]]:
    """
    Create and evaluate baseline models for comparison.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of baseline model evaluation results
    """
    baselines = {
        "mean": BaselineEstimator(strategy="mean"),
        "median": BaselineEstimator(strategy="median"),
        "last": BaselineEstimator(strategy="last"),
        "trend": BaselineEstimator(strategy="trend"),
    }
    
    results = {}
    
    for name, baseline in baselines.items():
        # Fit baseline
        baseline.fit(X_train, y_train)
        
        # Predict
        y_pred = baseline.predict(X_test)
        
        # Evaluate
        evaluator = FinancialRegressionEvaluator()
        metrics = evaluator.evaluate(y_test.to_numpy(), y_pred)
        
        results[name] = {
            "metrics": metrics,
            "model": baseline
        }
    
    return results


def create_production_pipeline(
    model: CatBoostQuantileRegressor,
    calibrator: Optional[QuantileConformalCalibrator] = None,
    postprocessor: Optional[CompositePredictionProcessor] = None
) -> Pipeline:
    """
    Assemble a complete production pipeline.
    
    Args:
        model: Fitted base model
        calibrator: Optional calibration pipeline
        postprocessor: Optional post-processing pipeline
        
    Returns:
        Complete production pipeline
    """
    steps = [('model', model)]
    
    if calibrator is not None:
        steps.append(('calibrator', calibrator))
    
    if postprocessor is not None:
        steps.append(('postprocessor', postprocessor))
    
    return Pipeline(steps)


def save_model_artifacts(
    model: CatBoostQuantileRegressor,
    evaluation_results: Dict[str, Any],
    output_path: Path,
    model_name: str = "catboost_quantile_regressor"
) -> Dict[str, Path]:
    """
    Save model and evaluation artifacts.
    
    Args:
        model: Fitted model to save
        evaluation_results: Evaluation results to save
        output_path: Directory to save artifacts
        model_name: Base name for saved files
        
    Returns:
        Dictionary mapping artifact types to file paths
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    artifact_paths = {}
    
    # Save model
    import joblib
    model_path = output_path / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    artifact_paths["model"] = model_path
    
    # Save evaluation results
    import json
    results_path = output_path / f"{model_name}_evaluation.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        else:
            return obj
    
    results_serializable = convert_types(evaluation_results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    artifact_paths["evaluation"] = results_path
    
    return artifact_paths


def save_predictions(
    predictions: np.ndarray,
    quantiles: List[float],
    output_path: Path,
    filename: str = "test_predictions.parquet"
) -> Path:
    """
    Save model predictions to file.
    
    Args:
        predictions: Quantile predictions array
        quantiles: List of quantile levels
        output_path: Directory to save predictions
        filename: Output filename
        
    Returns:
        Path to saved predictions file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create predictions DataFrame
    predictions_df = pl.DataFrame({
        f"q{int(q*100)}": predictions[:, i] 
        for i, q in enumerate(quantiles)
    })
    
    predictions_path = output_path / filename
    predictions_df.write_parquet(predictions_path)
    
    return predictions_path


def save_configuration(
    config_dict: Dict[str, Any],
    output_path: Path,
    filename: str = "modeling_config.json"
) -> Path:
    """
    Save modeling configuration to file.
    
    Args:
        config_dict: Configuration dictionary to save
        output_path: Directory to save configuration
        filename: Output filename
        
    Returns:
        Path to saved configuration file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_path = output_path / filename
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return config_path